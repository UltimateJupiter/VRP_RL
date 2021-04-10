import torch
from env.utils import log

class VRPReward():

    def __init__(self, wait_cost=0, opr_cost=0, time_penalty_power=1, bus_efficiency_reward=0, invalid_route_cost=0):
        self.wait_cost = wait_cost
        self.opr_cost = opr_cost
        self.time_penalty_power = time_penalty_power
        self.bus_efficiency_reward = bus_efficiency_reward
        self.invalid_route_cost = invalid_route_cost

        log("Initialize reward function | WaitCost:{} | OprCost:{} | TimePenaltyPower:{} | BusEfficiency:{} | InvalidRouteCost:{}".format(self.wait_cost, self.opr_cost, self.time_penalty_power, self.bus_efficiency_reward, self.invalid_route_cost))

    def rewards_terms(self, vec):
        vec_bus, vec_station, t = vec
        opr_reward = torch.Tensor.sum(vec_bus[:, 0]) * self.opr_cost
        queue_length, queue_time = vec_station[:, :, 0], vec_station[:, :, 1:]

        time_penalty_station = torch.sum(torch.pow(queue_time, self.time_penalty_power), dim=-1) # pylint:disable=no-member
        wait_reward = torch.sum(queue_length * time_penalty_station) * self.wait_cost # pylint:disable=no-member
        
        bus_capacity = torch.Tensor.sum(vec_bus[:, 0] * vec_bus[:, 3])
        if bus_capacity > 0:
            bus_efficiency = torch.Tensor.sum(vec_bus[:, 4:]) / bus_capacity
        else:
            bus_efficiency = 0
        return wait_reward, opr_reward, torch.Tensor.sum(queue_length), bus_efficiency * self.bus_efficiency_reward, bus_efficiency
    
    def __call__(self, vec, n_invalid_route, reward_verbose=False, info='', **kwargs):
        wait_reward, opr_reward, queue, bus_efficiency, _ = self.rewards_terms(vec)
        reward = wait_reward + opr_reward + bus_efficiency + n_invalid_route * self.invalid_route_cost
        if reward_verbose:
            print("{}TotalR: {:.5g} | Wait+: {:.4g} | Operation+: {:.4g} | Efficiency+: {:.4g} | InvalidRoutes: {}".format(info, reward, wait_reward, opr_reward, bus_efficiency, n_invalid_route))
        return reward