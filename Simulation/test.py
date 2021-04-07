from environ import duke_simple
from model.vis import *

# ev = UniRandEvent(1, np.array([200, 0, 1000]), np.array([]))

# for i in range(100):
#     print(ev.get_flow(i))

import numpy as np
np.random.seed(1)

M = duke_simple(n_bus=4, device='cpu', skip_station=True, verbose=True)
print(M.routes)

for t in range(120):
    M.step()
    # M.print_status()
M.schedule([0, 1, 2, 3])
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