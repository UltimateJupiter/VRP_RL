from model.playground import Map, Station, Bus
from model.events import UniRandEvent
import numpy as np

# ev = UniRandEvent(1, np.array([200, 0, 1000]), np.array([]))

# for i in range(100):
#     print(ev.get_flow(i))

import numpy as np

dukev1 = Map("./map_simp.csv", 'cpu')

route = dukev1.routes[0]
print(route)
bus1 = Bus(dukev1, 1, 20)
bus1.get_route(route)
ev1 = UniRandEvent(dukev1, 0, np.array([0, 50, 20, 40, 100]))
s1 = dukev1.stations[0]
s1.add_event(ev1)
# dukev1.plot_simple()
for t in range(10):
    dukev1.step()
    s1.get_flow()
    bus1.print_status()
    bus1.step()
    print(s1.discrete_queue)
exit()