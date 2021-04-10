from environ import duke_simple, duke_v1
from random_agents import UniRandAgent
from env.vis import *

from reward import VRPReward

from joblib import Parallel, delayed

from copy import deepcopy
# ev = UniRandEvent(1, np.array([200, 0, 1000]), np.array([]))

# for i in range(100):
#     print(ev.get_flow(i))

import numpy as np
np.random.seed(1)

# M = duke_simple(n_bus=1,e device='cpu', verbose=False)

reward_rule = VRPReward(wait_cost=0.1, opr_cost=10, bus_efficiency_scale=1000)

M = duke_v1(n_bus=15, device='cpu', reward_rule=reward_rule, verbose=False)
A = UniRandAgent(M)
# M_copy = deepcopy(M)
M.assign_auto_agent(A)
# print(M.routes)
vecs = []
print(M.state_dim, M.action_dim)
vec = M.vec_flatten
print(M.unflatten_vec(vec))
exit()
for t in range(10000):
    M.step_auto(reward_verbose=True)
    # exit()
    # vis_map(M, M.vec, background=False)

    # vecs.append(M.vec)
exit()
Parallel(n_jobs=8, verbose=10)(delayed(vis_map)(M, vec) for vec in vecs)

for vec in vecs:
    vis_map(M, vec)
exit()




exit()
for t in range(1200):
    M.step()
M.print_status()
exit()
M.schedule_buses([0, 1, 2, 3])
for t in range(20):
    M.step()
    M.print_status()
print(M.vec)
# for station in M.stations:
#     get_queue(station)
# exit()
# Map("./map_simp.csv", n_bus=4, device='cpu', skip_station=True)
# # dukev1 = Map("./map_v1.csv", n_bus=4, device='cpu', skip_station=True)
# dukev1.plot_simple()
# np.random.seed(0)
# # dukev1.plot_simple()
# for t in range(3):
#     dukev1.step()
#     dukev1.print_status()
# exit()