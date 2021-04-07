from .maps import DukeMap # pylint:disable=relative-beyond-top-level
from .utils import log # pylint:disable=relative-beyond-top-level
import torch
import numpy as np

class Map(DukeMap):
    def __init__(self, csv_fl, n_bus, device, skip_station=False, verbose=False):
        super().__init__(csv_fl, skip_station)
        self.t = 0
        self.dev = device
        self.total_t = 24 * 60 * 3 # 20 seconds per frame
        self.n_bus = n_bus
        self.stations = []
        self.buses = []
        self.verbose = verbose

        self.init_stations()
        self.init_buses()

    @property
    def n_station(self):
        return len(self.station_inds)

    def init_stations(self):
        for s_ind in range(self.n_station):
            node = self.nodes[s_ind]
            assert node.is_station
            station = Station(self, s_ind, node.name, verbose=self.verbose)
            station.print_status()
            self.stations.append(station)
    
    def init_buses(self):
        for b_ind in range(self.n_bus):
            bus = Bus(self, b_ind, 30, verbose=self.verbose)
            bus.print_status()
            self.buses.append(bus)
    
    def step(self):
        self.t += 1
        log("step: t={}".format(self.t))
        for station in self.stations:
            station.step()
        for bus in self.buses:
            if bus.active:
                bus.step()

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
    
    def schedule(self, bus_vec):
        assert len(bus_vec) == self.n_bus
        for i, bus in enumerate(self.buses):
            route_ind = bus_vec[i]
            if route_ind != -1:
                bus.get_route(route_ind)
    
    @property
    def vec(self):
        vec_bus = [bus.vec for bus in self.buses]
        vec_station = [station.vec for station in self.stations]
        return [vec_bus, vec_station, self.t]


class Bus():

    def __init__(self, M : Map, ind, capacity, init_location=0, stop_time_total=3, verbose=False):

        self.M = M
        self.dev = self.M.dev
        self.ind = ind
        self.capacity = capacity
        self.init_location = init_location
        self.stop_time_total = stop_time_total - 1

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

        assert not self.active
        route = self.M.routes[route_ind]
        
        path, stop_stations = route
        assert path[0] == self.location
        
        self.path = path
        self.stop_stations = stop_stations
        self.progress = 0
        self.stop_time = self.stop_time_total
        self.active = True
        if self.verbose:
            print("Bus {} gets route {} {}".format(self.ind, self.path, self.stop_stations))
    
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
        self.print_status()
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
        passenger_str = ' '.join(["s{}:{}".format(s, int(self.passengers[s])) for s in range(self.M.n_station)])
        print(" - Bus #{} - loc {} - active {} - {}".format(self.ind, self.location, self.active, passenger_str))

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
    def vec(self):
        info = [int(self.active), self.route_ind, int(self.location)]
        vec = torch.cat([torch.Tensor(info, device=self.dev), self.passengers]).float() # pylint: disable=no-member
        return vec


class Station():

    def __init__(self, M: Map, ind, name, discretize_slices=5, verbose=False):
        self.ind = ind
        self.M = M
        self.name = name
        self.dev = self.M.dev
        self.discretize_slices = discretize_slices
        self.events = []

        self.queue = [[] for s in range(self.M.n_station)]
        
        self.verbose = verbose
    
    def add_event(self, event):
        self.events.append(event)

    def refresh_events(self):
        for event in self.events:
            event.refresh()

    def get_flow(self):
        t = self.M.t
        flow = np.zeros(self.M.n_station)
        for event in self.events:
            flow += event.get_flow(t)
        for i in range(self.M.n_station):
            if flow[i] != 0:
                self.queue[i].append([t, flow[i]])

    def no_passenger(self, target_stations):
        k = sum([len(self.queue[s]) for s in target_stations])
        return k == 0

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

        if self.no_passenger(target_stations):
            return
        
        new_passengers = torch.zeros(self.M.n_station, device=self.dev) # pylint:disable=no-member
        t_start = 1e10
        for s in target_stations:
            if len(self.queue[s]) > 0:
                t_start = min(t_start, self.queue[s][0][0])
        
        for t_priority in range(t_start, self.M.t + 1):
            for s in target_stations:
                if len(self.queue[s]) == 0:
                    continue
                if self.queue[s][0][0] == t_priority:
                    if remain_seats >= self.queue[s][0][1]:
                        # All passengers get on the bus
                        new_passengers[s] += self.queue[s][0][1]
                        remain_seats -= self.queue[s][0][1]
                        self.queue[s] = self.queue[s][1:]
                    else:
                        self.queue[s][0][1] -= remain_seats
                        remain_seats = 0
                if remain_seats == 0:
                    break
            if remain_seats == 0:
                break
        bus.passengers += new_passengers

    def print_status(self):
        passenger_str = ' '.join(["s{}:{}".format(s, int(self.vec[0][s])) for s in range(self.M.n_station)])
        print(" *  Station #{} {}\t- {}".format(self.ind, self.name, passenger_str))

    def step(self):
        self.get_flow()

    def print_events(self):
        print("Station #{} {} Events:".format(self.ind, self.name))
        for ev in self.events:
            print("  ", ev)

    def reset(self):
        self.queue = [[] for s in range(self.M.n_station)]

    @property
    def vec(self):
        current_time = self.M.t
        queue_length = torch.zeros((self.M.n_station), device=self.dev) # pylint: disable=no-member
        wait_time = torch.zeros((self.M.n_station, self.discretize_slices), device=self.dev) # pylint: disable=no-member
        
        for s in range(self.M.n_station):
            count = 0
            for [t, p] in self.queue[s]:
                count += p
            queue_length[s] += count
        
        for s in range(self.M.n_station):
            marks = np.linspace(0, queue_length[s], self.discretize_slices)
            marked = 0
            count = 0
            for [t, p] in self.queue[s]:
                count += p
                if count >= marks[0]:
                    wait_time[s][marked:] = current_time - t
                    marked += 1
                    marks = marks[1:]
        return [queue_length, wait_time]


class VRP_Agent():

    def __init__(self, M : Map):
        self.M = M
    
    def schedule(self, info):
        bus_vec = - torch.ones(self.M.n_bus)
        self.M.schedule(bus_vec)
