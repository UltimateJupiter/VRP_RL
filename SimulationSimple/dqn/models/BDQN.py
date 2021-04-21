import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical 


class DuelingNetwork(nn.Module): 

    def __init__(self, obs, ac): 

        super().__init__()

        self.model = nn.Sequential(nn.Linear(obs, 512), 
                                   nn.ReLU(), 
                                   nn.Linear(512, 256), 
                                   nn.ReLU(), 
                                   nn.Linear(256, 256), 
                                   nn.ReLU())

        self.value_head = nn.Linear(128, 1)
        self.adv_head = nn.Linear(128, ac)

    def forward(self, x): 

        out = self.model(x)

        value = self.value_head(out)
        adv = self.adv_head(out)

        q_val = value + adv - adv.mean(1).reshape(-1,1)
        return q_val


class BranchingQNetwork(nn.Module):

    def __init__(self, obs, ac_dim, n, l1w=512, l2w=256, l3w=256, **kwargs): 

        super().__init__()

        self.ac_dim = ac_dim
        self.n = n 

        self.model = nn.Sequential(nn.Linear(obs, l1w), 
                                   nn.ReLU(),
                                   nn.Linear(l1w,l2w), 
                                   nn.ReLU(),
                                   nn.Linear(l2w,l3w), 
                                   nn.ReLU())

        self.value_head = nn.Linear(l3w, 1)
        self.adv_heads = nn.ModuleList([nn.Linear(l3w, n) for i in range(ac_dim)])

    def forward(self, x): 

        out = self.model(x)
        value = self.value_head(out)
        advs = torch.stack([l(out) for l in self.adv_heads], dim=1)
        q_val = value.unsqueeze(2) + advs - advs.mean(2, keepdim=True)
        # input(q_val.shape)
        return q_val

    def adv_gradient_rescale(self):
        for layer in self.adv_heads:
            for param in layer.parameters():
                param.grad /= (1 + self.ac_dim)
# b = BranchingQNetwork(5, 4, 6)

# b(torch.rand(10, 5))