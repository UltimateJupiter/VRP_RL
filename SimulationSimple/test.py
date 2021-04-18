from environ import duke_simple, duke_v1, make_environ
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

reward_rule = VRPReward()
M = make_environ(reward_rule, map_dirc='./config/3s_1l_exp')
A = UniRandAgent(M)
M.assign_auto_agent(A)
vecs = []
for t in range(10000):
    M.step_auto(reward_verbose=True)
    # exit()
    vis_map(M, M.vec, background=False, margin=1)
    print(t)

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