import matplotlib as plt
from .playground import Map, Station, Bus
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def aggregate_flow(flow, density):
    density = 3 * density
    t_all = len(flow)

    time = np.arange(max(t_all // density + 1, 1)) / (3 * 60) * density
    res = np.zeros_like(time)
    for t in range(t_all):
        res[t // density] += flow[t]
    width = density / (3 * 60)
    return time, res, width

def get_queue(s : Station):
    queue_array = np.zeros([len(s.queue), s.M.t + 1])
    for target_ind in range(s.M.n_station):
        for time, flow in s.queue[target_ind]:
            queue_array[target_ind][time] = flow
    return queue_array

def vis_flow(M : Map, density=5):
    n = M.n_station
    fig = plt.figure(constrained_layout=True, figsize=(10, 10))
    spec = gridspec.GridSpec(ncols=n, nrows=n, figure=fig)
    for i, station in enumerate(M.stations):
        flows = get_queue(station)
        for j, flow in enumerate(flows):
            ax = fig.add_subplot(spec[i, j])
            time, res, width = aggregate_flow(flow, density)
            ax.bar(time, res, width)
            if i == n - 1:
                ax.set_xlabel(M.stations[j].name)
            if j == 0:
                ax.set_ylabel(M.stations[i].name)
    plt.savefig("flow.jpg")
