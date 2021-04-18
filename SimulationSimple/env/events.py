import torch
import numpy as np
from numpy.random import standard_normal
from scipy.stats import norm
from scipy.special import erfc #pylint:disable=no-name-in-module

import matplotlib.pyplot as plt

def CDF_diffusion(R, d, t):
    return erfc(R / (2 * np.sqrt(d * (t + 1e-18))))

def diffusion_flow(mean : float, dist : float, flow : float, time : int, span=180):
    if time < mean or time > mean + span:
        return 0
    D = 1
    t = (time - mean)
    normalizing_factor = CDF_diffusion(dist, D, span)
    E_flow = flow * (CDF_diffusion(dist, D, t + 1) - CDF_diffusion(dist, D, t)) / normalizing_factor
    return np.random.poisson(E_flow)

class DiffEvent():
    def __init__(self, M, self_ind, target_ind, t_peak_mean : int, dist_mean : int, flow_mean : int, t_peak_coef_std=10, dist_coef_std=0.2, flow_coef_std=0.2, span=4):
        self.self_ind = self_ind
        self.target_ind = target_ind
        self.M = M
        assert self.self_ind != self.target_ind

        self.t_peak_mean = t_peak_mean * 60 * M.frame_rate
        self.dist_mean = dist_mean / M.frame_rate * M.frame_rate
        self.flow_mean = flow_mean

        self.t_peak_coef_std = t_peak_coef_std
        self.dist_coef_std = dist_coef_std * dist_mean
        self.flow_coef_std = flow_coef_std * flow_mean

        self.t_peak, self.dist, self.flow = -1, -1, -1
        self.span = span * M.frame_rate * 60
        self.reset()
    
    def __str__(self):
        return "DiffRandEV {} -> {} | Tpeak {:.4g}  Tspan {:.4g}  Flow {:.4g}".format(self.self_ind, self.target_ind, self.t_peak/60, self.dist, self.flow)
    
    def reset(self):
        self.t_peak = self.t_peak_mean + standard_normal() * self.t_peak_coef_std
        self.dist = self.dist_mean + standard_normal() * self.dist_coef_std
        self.flow = self.flow_mean + standard_normal() * self.flow_coef_std
        self.dist *= (self.dist >= 0)
        self.flow *= (self.flow >= 0)
        return

    def get_flow(self, time=None):
        if time is None:
            time = self.M.t
        if self.M is not None:
            flow = np.zeros(self.M.n_station)
        else:
            flow = np.zeros(2)
        flow[self.target_ind] = diffusion_flow(self.t_peak, self.dist, self.flow, time, self.span)
        return flow

class UniRandEvent():

    def __init__(self, M, self_ind : int, weights : np.array, coef_std=0.1):
        """Uniformly Random Event

        Parameters
        ----------
        self_ind : int
            index of the station
        weights : np.array
            flow to other stations (person / hour)
        """
        self.M = M
        self.self_ind = self_ind
        if M is not None:
            assert len(weights) == M.n_station
        self.weights_init = weights / 60 / M.frame_rate # normalize weight to each frame
        assert self.weights_init[self_ind] == 0 # There should be no flow to itself

        self.coef_std = coef_std
        self.weights = np.zeros_like(self.weights_init)
        self.reset()

    def reset(self):
        self.weights = self.weights_init + self.weights_init * np.random.normal(scale=self.coef_std, size=len(self.weights_init))
        self.weights *= (self.weights >= 0)

    def get_flow(self, time=None):
        if time is None:
            time = self.M.t
        return np.random.poisson(self.weights)

    def __str__(self):
        return "UniRandEV {} | weights {}".format(self.self_ind, self.weights)

class GaussRandEvent():

    def __init__(self, M, self_ind : int, weights : np.array, center : float, std : float, coef_std=0.1):

        self.M = M
        self.self_ind = self_ind
        self.weights_init = weights / 60 / M.frame_rate # normalize weight to each frame
        self.centers_init = center * np.ones_like(self.weights_init) * M.frame_rate * 60
        self.stds_init = std * np.ones_like(self.weights_init) * M.frame_rate * 60

        self.coef_std = coef_std

        self.weights = np.zeros_like(self.weights_init)
        self.centers = np.zeros_like(self.weights_init)
        self.stds = np.zeros_like(self.weights_init)

        if M is not None:
            assert len(weights) == M.n_station
        assert self.weights[self_ind] == 0 # There should be no flow to itself

        self.reset()

    def get_flow(self, time : int):
        if time is None:
            time = self.M.t
        flow = np.exp(- np.power(time - self.centers, 2) / (2 * np.power(self.stds, 2)))
        return np.random.poisson(self.weights * flow)
    
    def reset(self):
        self.weights = self.weights_init + self.weights_init * np.random.normal(scale=self.coef_std, size=len(self.weights_init))
        self.weights *= (self.weights >= 0)
        self.centers = self.centers_init + self.centers_init * np.random.normal(scale=self.coef_std, size=len(self.weights_init))
        self.stds = self.stds_init + self.stds_init * np.random.normal(scale=self.coef_std, size=len(self.weights_init))
    
    def __str__(self):
        return "GaussianRandEV {} | center {:.4g} | std {:.4g} | weights {}".format(self.self_ind, self.centers[0] / 60 / self.M.frame_rate, self.stds[0] / 60 / self.M.frame_rate, self.weights)

class IterEvent():

    def __init__(self, M, self_ind : int, target_ind : int, flow : int, begin : int, lapse : int, end=1e8):

        self.M = M
        self.self_ind = self_ind
        self.target_ind = target_ind
        self.flow = flow
        self.begin = begin
        self.end = end
        self.lapse = lapse
        self.reset()

    def get_flow(self, time : int):
        ret = np.zeros(self.M.n_station)
        if time is None:
            time = self.M.t
        if time >= self.begin and time < self.end:
            if time % self.lapse == self.begin % self.lapse:
                ret[self.target_ind] = np.random.poisson(self.flow)
        return ret
    
    def reset(self):
        return
    
    def __str__(self):
        return "IterEvent {} | flow {:.4g} | start {:.4g} | lapse {}".format(self.self_ind, self.centers[0] / 60 / self.M.frame_rate, self.stds[0] / 60 / self.M.frame_rate, self.weights)


def vis_event_diff():
    
    t_peak = 0.5
    t_dist = 10
    t_flow = 200
    ev = DiffEvent(None, 1, 0, t_peak, t_dist, t_flow)
    frame_rate = ev.M.frame_rate
    density = frame_rate * 1
    t_all = 5 * 60 * frame_rate
    time = np.arange(t_all // density) / (frame_rate * 60) * density
    res = np.zeros_like(time)
    for t in range(t_all):
        print(ev)
        res[t // density] += ev.get_flow(t)[0]
        print(ev.get_flow(t))
    print(time)
    plt.bar(time, res, width=1 / (frame_rate * 60) * density)
    plt.show()

if __name__ == "__main__":
    # vis_event_uni()
    # vis_event_gaussian()
    vis_event_diff()