import numpy as np
from model.events import DiffEvent, UniRandEvent, GaussRandEvent
from model.playground import Map
import os
import csv

def read_event_csv(M : Map, csv_fl):
    assert os.path.isfile(csv_fl)
    with open(csv_fl, newline='') as csvfile:
        lines = [x for x in csv.reader(csvfile)]
    for i in range(M.n_station):
        assert int(lines[i][0]) == i
        assert lines[i][1] == M.stations[i].name
    assert lines[M.n_station][0] == '***'
    assert lines[M.n_station + 2][0] == '***'
    return [[float(i) for i in x] for x in lines[M.n_station + 3:]]

def add_diff_event(M : Map, diff_csv):

    event_configs = read_event_csv(M, diff_csv)
    for s, d, time, dist, flow in event_configs:
        assert s <= M.n_station
        assert d <= M.n_station
        ev = DiffEvent(M, int(s), int(d), time, dist, flow)
        M.stations[int(s)].add_event(ev)

def add_uniform_event(M : Map, uniform_csv):
    
    event_configs = read_event_csv(M, uniform_csv)
    for event_config in event_configs:
        assert len(event_config) == M.n_station + 1
        s = int(event_config[0])
        assert s <= M.n_station
        weights = np.array(event_config[1:])
        assert sum(weights >= 0) == M.n_station
        ev = UniRandEvent(M, s, weights)
        M.stations[int(s)].add_event(ev)

def add_gaussian_event(M : Map, gaussian_csv):

    event_configs = read_event_csv(M, gaussian_csv)
    for event_config in event_configs:
        assert len(event_config) == M.n_station + 3
        s = int(event_config[0])
        center, std = event_config[1: 3]
        assert s <= M.n_station
        weights = np.array(event_config[3:])
        assert sum(weights >= 0) == M.n_station
        ev = GaussRandEvent(M, s, weights, center, std)
        M.stations[int(s)].add_event(ev)

def get_events(M : Map, dirc):
    assert os.path.isdir(dirc)

    diff_csv = os.path.join(dirc, 'diffusion_flow.csv')
    add_diff_event(M, diff_csv)

    uniform_csv = os.path.join(dirc, 'uniform_flow.csv')
    add_uniform_event(M, uniform_csv)

    gaussian_csv = os.path.join(dirc, 'gaussian_flow.csv')
    add_gaussian_event(M, gaussian_csv)

def make_environ(dirc, n_bus, device, skip_station, **kwargs):

    map_csv = os.path.join(dirc, 'map.csv')

    M = Map(map_csv, n_bus, device, skip_station, **kwargs)
    events_dirc = os.path.join(dirc, 'events')
    get_events(M, events_dirc)

    return M

def duke_simple(n_bus, device, skip_station, **kwargs):
    dirc = './config/simple'
    return make_environ(dirc, n_bus, device, skip_station, **kwargs)