import torch
import numpy as np
from env.utils import log

class VRPReward():

    def __init__(self, queue_reward=0, wait_reward=0, opr_reward=0, time_penalty_power=1, efficiency_reward=0, invalid_route_reward=0):
        self.queue_reward = queue_reward
        self.wait_reward = wait_reward
        self.opr_reward = opr_reward
        self.time_penalty_power = time_penalty_power
        self.efficiency_reward = efficiency_reward
        self.invalid_route_reward = invalid_route_reward

        log("Initialize reward function | QueueCost:{} | WaitCost:{} | OprCost:{} | TimePenaltyPower:{} | BusEfficiency:{} | InvalidRouteCost:{}".format(self.queue_reward, self.wait_reward, self.opr_reward, self.time_penalty_power, self.efficiency_reward, self.invalid_route_reward))

    def rewards_terms(self, vec):
        vec_bus, vec_station, t = vec
        opr = torch.Tensor.sum(vec_bus[:, 0])
        queue_length, queue_time = vec_station[:, :, 0], vec_station[:, :, 1:]

        time_penalty_station = torch.sum(torch.pow(queue_time, self.time_penalty_power), dim=-1)
        wait = torch.Tensor.sum(queue_length * time_penalty_station)
        
        bus_capacity = torch.Tensor.sum(vec_bus[:, 0] * vec_bus[:, 3])
        if bus_capacity > 0:
            efficiency = torch.Tensor.sum(vec_bus[:, 4:]) / bus_capacity
        else:
            efficiency = 0

        queue = torch.Tensor.sum(queue_length)
        info = [queue, wait, opr, efficiency]
        return info
    
    def __call__(self, vec, n_invalid_route, reward_verbose=False, info='', **kwargs):
        queue, wait, opr, efficiency = self.rewards_terms(vec)
        reward = wait * self.wait_reward \
               + queue * self.queue_reward \
               + opr * self.opr_reward \
               + efficiency * self.efficiency_reward \
               + n_invalid_route * self.invalid_route_reward
        return reward, np.array([queue.item(), wait.item(), opr.item(), efficiency.item(), n_invalid_route])