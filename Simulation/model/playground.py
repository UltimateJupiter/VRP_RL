from .maps import DukeMap # pylint:disable=relative-beyond-top-level
import torch
import numpy as np

class Map(DukeMap):
    def __init__(self, csv_fl, device, skip_station=False):
        super().__init__(csv_fl, skip_station)
        self.t = 0
        self.dev = device
        self.total_t = 24 * 60 * 3 # 20 seconds per frame
        self.stations = []
        for s_ind in range(self.station_count):
            node = self.nodes[s_ind]
            assert node.is_station
            station = Station(self, s_ind, node.name)
            station.print_status()
            self.stations.append(station)
        self.vertices = []
        return

    @property
    def station_count(self):
        return len(self.station_inds)
    
    def step(self):
        self.t += 1
        print("Step! t={}".format(self.t))

    def refresh(self):
        for station in self.stations:
            station.refresh_events()
    
    def reset(self):
        self.t = 0


class Bus():

    def __init__(self, M : Map, ind, capacity, init_location=0, stop_time_total=3):

        self.M = M
        self.dev = self.M.dev
        self.ind = ind
        self.capacity = capacity
        self.stop_time_total = stop_time_total

        self.active = False

        self.passengers = torch.zeros(M.station_count, device=self.dev)
        self.path = None
        self.stop_stations = None

        self.location = init_location
        self.progress = -1
        self.stop_time = -1

    def get_route(self, route):

        assert not self.active
        
        path, stop_stations = route
        assert path[0] == self.location
        
        self.path = path
        self.stop_stations = stop_stations
        self.progress = 0
        self.stop_time = self.stop_time_total
        self.active = True
        print("Bus {} gets route {} {}".format(self.ind, self.path, self.stop_stations))
    
    @property
    def passenger_count(self):
        return torch.Tensor.sum(self.passengers)
    
    @property
    def remain_seats(self):
        return self.capacity - self.passenger_count
    
    def step(self):
        assert self.active == True
        print(self.active)
        if self.stop_time > 0:
            print(" * * Bus{} waiting, {} frames remaining".format(self.ind, self.stop_time))
            self.stop_time -= 1

        else:
            self.progress += 1
            self.location = self.path[self.progress]
            print(" * * Bus{} moves to node {}".format(self.ind, self.location))

            if self.location in self.stop_stations:
                self.interact_station(self.M.stations[self.location])
                self.stop_time = self.stop_time_total
        
        if self.location == self.path[-1]:
            print(" * * Bus{} completed route".format(self.ind))
            self.deactivate()

    def interact_station(self, station):
        assert self.M.nodes[self.location].is_station
        station.interact_bus(self)
    
    def deactivate(self):
        assert self.passenger_count == 0
        self.active = False
        self.path = None
        self.stop_stations = None
        self.progress = -1
        return
    
    def print_status(self):
        passenger_str = ' '.join(["s{}:{}".format(s, int(self.passengers[s])) for s in range(self.M.station_count)])
        print(" - Bus #{} - loc {} - active {} - {}".format(self.ind, self.location, self.active, passenger_str))

    def update(self):
        self.location = 1
        

class Station():

    def __init__(self, M: Map, ind, name, discretize_slices=5):
        self.ind = ind
        self.M = M
        self.name = name
        self.dev = self.M.dev
        self.discretize_slices = discretize_slices
        self.events = []

        self.queue = [[] for s in range(self.M.station_count)]
    
    def add_event(self, event):
        self.events.append(event)

    def refresh_events(self):
        for event in self.events:
            event.refresh()

    def get_flow(self):
        t = self.M.t
        flow = np.zeros(self.M.station_count)
        for event in self.events:
            flow += event.get_flow(t)
        for i in range(self.M.station_count):
            if flow[i] != 0:
                self.queue[i].append([t, flow[i]])
        print(self.queue)

    def no_passenger(self, target_stations):
        k = sum([len(self.queue[s]) for s in target_stations])
        return k == 0

    def interact_bus(self, bus : Bus):
        # Get off all passengers
        print(" * * Station {} {} interacting with bus {}".format(self.ind, self.name, bus.ind))
        bus.passengers[self.ind] = 0

        target_stations = bus.stop_stations
        remain_seats = bus.remain_seats

        if self.no_passenger(target_stations):
            return
        
        new_passengers = torch.zeros(self.M.station_count, device=self.dev) # pylint:disable=no-member
        t_start = 1e10
        for s in target_stations:
            if len(self.queue[s]) > 0:
                t_start = min(t_start, self.queue[s][0][0])
        
        for t_priority in range(t_start, self.M.t):
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
        passenger_str = ' '.join(["s{}:{}".format(s, int(self.discrete_queue[0][s])) for s in range(self.M.station_count)])
        print(" *  Station #{} {}\t- {}".format(self.ind, self.name, passenger_str))


    @property
    def discrete_queue(self):
        current_time = self.M.t
        queue_length = torch.zeros((self.M.station_count), device=self.dev) # pylint: disable=no-member
        wait_time = torch.zeros((self.M.station_count, self.discretize_slices), device=self.dev) # pylint: disable=no-member
        
        for s in range(self.M.station_count):
            count = 0
            for [t, p] in self.queue[s]:
                count += p
            queue_length[s] += count
        
        for s in range(self.M.station_count):
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
