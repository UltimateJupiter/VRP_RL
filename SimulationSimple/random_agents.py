from env.playground import Map
import torch
import numpy as np

class UniRandAgent():

    def __init__(self, M : Map):
        self.M = M
        return

    def action(self, inputs):
        schedule = [-1] * self.M.n_bus
        # pylint:disable=no-member
        vec_bus, vec_station, _ = inputs
        for i, bus_info in enumerate(vec_bus):
            active, _, loc = bus_info[:3]
            if active:
                continue
            else:
                r_ind = np.random.choice(self.M.routes_choice[loc.item()])
                schedule[i] = r_ind
        # print(vec_bus)
        return schedule