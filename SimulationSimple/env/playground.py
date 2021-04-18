import torch
import numpy as np
from .maps import DukeMap
from .utils import log

from datetime import datetime
from joblib import Parallel, delayed

class Map(DukeMap):
    def __init__(self,
                 csv_fl,
                 device,
                 reward_rule,
                 n_bus=1,
                 bus_capacity=60,
                 bus_stop_time=1,
                 frame_rate=3,
                 skip_station=False,
                 verbose=False,
                 total_time=12,
                 **kwargs):
        
        super().__init__(csv_fl, frame_rate, skip_station)
        self.t = 0
        self.dev = device
        self.total_t = total_time * 60 * self.frame_rate
        
        self.n_bus = n_bus
        self.n_station = len(self.station_inds)

        self.stations = []
        self.buses = []
        self.verbose = verbose
        self.reward_rule = reward_rule

        self.init_stations()
        self.init_buses(bus_capacity, bus_stop_time)

        self.n_routes = len(self.routes)
        self.n_station = len(self.stations)
        self.auto_agent = None

        self.state_dim = len(self.vec_flatten)
        self.state_dims = [x.shape for x in self.vec[:-1]]
        self.action_dim = [self.n_bus, self.n_routes + 1]

    def init_stations(self):
        for s_ind in range(self.n_station):
            node = self.nodes[s_ind]
            assert node.is_station
            station = Station(self, s_ind, node.name, node, verbose=self.verbose)
            # station.print_status()
            self.stations.append(station)
    
    def init_buses(self, capacity, stop_time):
        for b_ind in range(self.n_bus):
            bus = Bus(self, b_ind, capacity, stop_time_total=stop_time, verbose=self.verbose)
            # bus.print_status()
            self.buses.append(bus)

    def refresh(self):
        for station in self.stations:
            station.refresh_events()
    
    def reset(self):
        self.t = 0
        for bus in self.buses:
            bus.reset()
        for station in self.stations:
            station.reset()

    def print_status(self):
        for station in self.stations:
            station.print_status()
        for bus in self.buses:
            bus.print_status()
    
    def print_events(self):
        for station in self.stations:
            station.print_events()
    
    def schedule_buses(self, bus_vec):
        n_invalid_route = 0
        assert len(bus_vec) == self.n_bus
        for i, bus in enumerate(self.buses):
            route_ind = bus_vec[i]
            valid = bus.get_route(route_ind)
            if valid != 1:
                n_invalid_route += 1
        return n_invalid_route
    
    def assign_auto_agent(self, agent):
        self.auto_agent = agent
        log("Agent Assigned {}".format(agent))

    @property
    def vec(self):
        vec_bus = torch.cat([bus.vec.unsqueeze(0) for bus in self.buses]) # pylint:disable=no-member
        vec_station = torch.cat([station.vec.unsqueeze(0) for station in self.stations]) # pylint:disable=no-member
        res = [vec_bus, vec_station, self.t]
        return res

    @property
    def vec_flatten(self):
        vec_bus, vec_station, t = self.vec
        res = torch.cat([vec_bus.view(-1), vec_station.view(-1), torch.Tensor([t]).to(self.dev)]) # pylint:disable=no-member
        return res

    def unflatten_vec(self, state):
        bus_len = self.n_bus * (4 + self.n_station)
        vec_bus, vec_station, t = state[:bus_len], state[bus_len:-1], state[-1]
        vec_bus = vec_bus.view(self.state_dims[0])
        vec_station = vec_station.view(self.state_dims[1])
        return [vec_bus, vec_station, t]

    def reward(self, n_invalid_route, **kwargs):
        return self.reward_rule(self.vec, n_invalid_route, **kwargs)

    def env_step(self, model_verbose=False, **kwargs):
        self.t += 1
        if model_verbose:
            log("step: t={}".format(self.t))
            
        for station in self.stations:
            station.step()
        for bus in self.buses:
            if bus.active:
                bus.step()
    
    def step_auto(self, **kwargs):
        assert self.auto_agent is not None
        bus_schedule = self.auto_agent.action(self.vec)
        self.schedule_buses(bus_schedule)
        self.env_step(**kwargs)
    
    def step(self, action, **kwargs):
        self.schedule_buses(action)
        self.env_step(**kwargs)
    
    def feedback_step(self, action, **kwargs):
        n_invalid_route = self.schedule_buses(action)
        self.env_step(**kwargs)
        done = self.t >= self.total_t
        reward, feedback, scale = self.reward(n_invalid_route, **kwargs)
        next_state = self.vec_flatten

        return next_state, reward, done, feedback, scale
        

class Bus():

    def __init__(self, M : Map, ind, capacity, init_location=0, stop_time_total=1, verbose=False):
        self.M = M
        self.dev = self.M.dev
        self.ind = ind
        self.capacity = capacity
        self.init_location = init_location
        self.stop_time_total = int(np.round(stop_time_total * M.frame_rate) - 1)

        self.active = False

        self.passengers = torch.zeros(M.n_station, device=self.dev) # pylint: disable=no-member
        self.path = None
        self.stop_stations = None

        self.location = self.init_location
        self.progress = -1
        self.stop_time = -1
        self.route_ind = -1

        self.verbose = verbose

    def get_route(self, route_ind):
        
        if route_ind == self.M.n_routes:
            if self.verbose:
                print("Bus {} stays idle".format(self.ind))
            return 1
        route_ind = int(route_ind)

        route = self.M.routes[route_ind]
        path, stop_stations = route

        if path[0] != self.location or self.active:
            if self.verbose:
                print("Bus {} gets invalid route".format(self.ind))
            return 0

        self.route_ind = route_ind
        
        self.path = path
        self.stop_stations = stop_stations
        self.progress = 0
        self.stop_time = self.stop_time_total
        self.active = True
        if self.verbose:
            print("Bus {} gets route {} {}".format(self.ind, self.path, self.stop_stations))
        return 1

    @property
    def passenger_count(self):
        return torch.Tensor.sum(self.passengers)
    
    @property
    def remain_seats(self):
        return self.capacity - self.passenger_count
    
    def step(self):
        assert self.active == True
        if self.stop_time > 0:
            if self.verbose:
                print(" * * Bus{} waiting, {} frames remaining".format(self.ind, self.stop_time))
            self.stop_time -= 1

        else:
            if self.location in self.stop_stations:
                self.interact_station_on(self.M.stations[self.location])
                self.stop_stations = self.stop_stations[1:]
            self.progress += 1
            self.location = self.path[self.progress]
            if self.location in self.stop_stations:
                self.interact_station_off(self.M.stations[self.location])
                self.stop_time = self.stop_time_total

            if self.verbose:
                print(" * * Bus{} moves to node {}".format(self.ind, self.location))
        if self.location == self.path[-1]:
            if self.verbose:
                print(" * * Bus{} completed route".format(self.ind))
            self.deactivate()

    def interact_station_on(self, station):
        assert self.M.nodes[self.location].is_station
        station.interact_bus_on(self, verbose=self.verbose)

    def interact_station_off(self, station):
        assert self.M.nodes[self.location].is_station
        station.interact_bus_off(self, verbose=self.verbose)

    def deactivate(self):
        # self.print_status()
        assert self.passenger_count == 0
        self.active = False
        self.path = None
        self.stop_stations = None
        self.progress = -1
        self.route_ind = -1
        if self.verbose:
            print(" * * Bus{} deactivated".format(self.ind))
        return
    
    def print_status(self):
        passenger_str = ' '.join(["{}".format(int(self.passengers[s])) for s in range(self.M.n_station)])
        print(" -  Bus #{} - active {} - loc {} - route {} - cap {} - [{}]".format(self.ind, self.active, self.location, self.route_ind, self.capacity, passenger_str))

    def reset(self):
        self.active = False

        self.passengers = torch.zeros(self.M.n_station, device=self.dev) # pylint: disable=no-member
        self.path = None
        self.stop_stations = None

        self.location = self.init_location
        self.progress = -1
        self.stop_time = -1
        self.route_ind = -1

    @property
    def crd(self):
        return self.M.nodes[self.location].crd

    @property
    def vec(self):
        info = [int(self.active), self.route_ind, int(self.location), self.capacity]
        vec = torch.cat([torch.Tensor(info).to(self.dev), self.passengers]).float() # pylint: disable=no-member
        return vec.to(self.dev)


class Station():

    def __init__(self, M: Map, ind, name, node, discretize_slices=5, verbose=False):
        self.ind = ind
        self.M = M
        self.name = name
        self.node = node
        self.dev = self.M.dev
        self.discretize_slices = discretize_slices
        self.events = []

        self.queue = torch.zeros(self.M.n_station)
        self.vec_cache = None
        self.vec_t = -1
        
        self.verbose = verbose
    
    def add_event(self, event):
        self.events.append(event)

    def refresh_events(self):
        for event in self.events:
            event.refresh()

    def get_flow(self):
        t = self.M.t
        flow = torch.zeros(self.M.n_station)
        for event in self.events:
            flow += event.get_flow(t)
        self.queue += flow

    def interact_bus_off(self, bus : Bus, verbose=False):
        # Get off all passengers
        if verbose:
            print(" * * Station {} {} interacting with bus {} (OFF)".format(self.ind, self.name, bus.ind))
        bus.passengers[self.ind] = 0

    def interact_bus_on(self, bus : Bus, verbose=False):
        if verbose:
            print(" * * Station {} {} interacting with bus {} (ON)".format(self.ind, self.name, bus.ind))

        target_stations = bus.stop_stations
        remain_seats = bus.remain_seats

        station_mask = torch.zeros(self.M.n_station)
        for s in target_stations:
            station_mask[s] = 1

        pendings = self.queue * station_mask
        if torch.sum(pendings) == 0:
            return
        new_passengers = torch.zeros(self.M.n_station) # pylint:disable=no-member
        if torch.sum(pendings) <= remain_seats:
            new_passengers += pendings

        else:
            new_passengers = torch.ceil(pendings / (sum(pendings) / remain_seats))
            diff = torch.sum(new_passengers) - remain_seats
            while diff > 0:
                choices = []
                for s in target_stations:
                    if new_passengers[s] > 0:
                        choices.append(s)
                ind = np.random.choice(choices)
                new_passengers[ind] -= 1
                diff = torch.sum(new_passengers) - remain_seats

        bus.passengers += new_passengers.to(self.M.dev)
        self.queue -= new_passengers

    def print_status(self):
        passenger_str = ' '.join(["s{}:{}".format(s, int(self.queue[s])) for s in range(self.M.n_station)])
        print(" *  Station #{} {}\t- {}".format(self.ind, self.name, passenger_str))

    def step(self):
        self.get_flow()

    def print_events(self):
        print("Station #{} {} Events:".format(self.ind, self.name))
        for ev in self.events:
            print("  ", ev)

    def reset(self):
        self.queue = torch.zeros(self.M.n_station)
        for event in self.events:
            event.reset()

    @property
    def vec(self, refresh=False):
        return self.queue.to(self.M.dev)

    @property
    def queue_length(self):
        return self.vec

