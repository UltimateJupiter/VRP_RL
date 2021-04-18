import random
import numpy as np
from collections import namedtuple
from env.utils import log
import torch

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):
    def __init__(self, capacity, device, state_size, n_action, hist_len):
        self.capacity = capacity
        self.device = device

        log("Initializing Episode Memory | Capacity {:.3g} | Requesting {:.4g}GB".format(capacity, 8 * capacity * (2 * state_size + n_action + 1) / (2 ** 30)))

        self.states = torch.zeros([capacity] + [state_size]).to(self.device)
        self.actions = torch.zeros([capacity] + [n_action], dtype=torch.long).to(self.device)
        self.rewards = torch.zeros([capacity]).to(self.device)
        self.next_states = torch.zeros([capacity] + [state_size]).to(self.device)

        self.hist_len = hist_len
        self.position = 0
        self.size = 0
        log("Episode Memory Initialized")

    def push(self, transition):
        """Saves a transition."""
        self.states[self.position] = transition.state.to(self.device)
        self.actions[self.position] = transition.action.to(self.device)
        self.rewards[self.position] = transition.reward.to(self.device)
        self.next_states[self.position] = transition.next_state.to(self.device)
        
        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size, out_device):
        indices = random.sample(range(self.size), batch_size)
        batch = Transition(
            self.states[indices].to(out_device),
            self.actions[indices].to(out_device),
            self.rewards[indices].to(out_device),
            self.next_states[indices].to(out_device))
        return batch

    def __len__(self):
        return self.size