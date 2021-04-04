import torch
import numpy as np
from numpy.random import standard_normal
from scipy.stats import norm
from scipy.special import erfc #pylint:disable=no-name-in-module

def CDF_diffusion(R, d, t):
    return erfc(R / (2 * np.sqrt(d * (t + 1e-18))))

def diffusion_flow(mean : float, dist : float, flow : float, time : int, span=180):
    if time < mean or time > mean + span:
        return 0
    D = 1
    t = (time - mean)
    normalizing_factor = CDF_diffusion(dist, D, 180)
    E_flow = flow * (CDF_diffusion(dist, D, t + 1) - CDF_diffusion(dist, D, t)) / normalizing_factor
    return np.random.poisson(E_flow)

class DiffEvent():

    def __init__(self, M, self_ind, target_ind, station_count, t_peak_mean : int, dist_mean : int, flow_mean : int, t_peak_std=10, dist_std=10, flow_std=0.1):
        self.self_ind = self_ind
        self.target_ind = target_ind
        self.M = M
        assert self.self_ind != self.target_ind

        self.t_peak_mean = t_peak_mean * 60
        self.dist_mean = dist_mean
        self.flow_mean = flow_mean

        self.t_peak_std = t_peak_std
        self.dist_std = dist_std
        self.flow_std = flow_std * flow_mean

        self.t_peak, self.dist, self.flow = -1, -1, -1
    
    def __str__(self):
        return "Tpeak {:.4g} Tspan {:.4g} Flow {:.4g}".format(self.t_peak, self.dist, self.flow)
    
    def refresh(self):
        self.t_peak = self.t_peak_mean + standard_normal() * self.t_peak_std
        self.dist = self.dist_mean + standard_normal() * self.dist_std
        self.flow = self.flow_mean + standard_normal() * self.flow_std
        return

    def get_flow(self, time=None):
        if time is None:
            time = self.M.t
        flow = np.zeros(self.M.station_count)
        flow[self.target_ind] = diffusion_flow(self.t_peak, self.dist, self.flow, time)
        return flow

class UniRandEvent():

    def __init__(self, M, self_ind : int, weights : np.array):
        """Uniformly Random Event

        Parameters
        ----------
        self_ind : int
            index of the station
        weights : np.array
            flow to other stations (person / hour)
        """
        self.M = M
        assert len(weights) == M.station_count
        self.weights = weights / 60 / 3 # normalize weight to each frame
        assert self.weights[self_ind] == 0 # There should be no flow to itself

    def get_flow(self, time=None):
        if time is None:
            time = self.M.t
        return np.random.poisson(self.weights)
    
    def refresh(self):
        return

class GaussRandEventAll():

    def __init__(self, M, self_ind : int, weights : np.array, centers : np.array, stds : np.array):
        """Gaussian Random Event

        Parameters
        ----------
        self_ind : int
            index of the station
        weights : np.array
            max flow to other stations (person / hour)
        centers : np.array
            time to reach max flow
        stds : np.array
            standard deviation
        """
        self.M = M
        self.weights = weights / 60 / 3 # normalize weight to each frame
        self.centers = centers
        self.stds = stds

        assert len(weights) == M.station_count
        assert self.weights[self_ind] == 0 # There should be no flow to itself

    def get_flow(self, time : int):
        if time is None:
            time = self.M.t
        flow = np.exp(- np.power(time - self.centers, 2) / (2 * np.power(self.stds, 2)))
        return np.random.poisson(self.weights * flow)
    
    def refresh(self):
        return

class Station():
    def __init__(self, name, neighbors, events):
        return
