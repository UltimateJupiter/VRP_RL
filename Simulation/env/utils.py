import datetime
from termcolor import colored

def powerset(s):
    res = []
    x = len(s)
    for i in range(1 << x):
        res.append([s[j] for j in range(x) if (i & (1 << j))])
    return res

def log(info, color='green', print_log=True):
    if print_log:
        # print(colored("[{}]  ", color).format(datetime.datetime.now()) + info)
        print("[{}]  ".format(datetime.datetime.now()) + info)
        