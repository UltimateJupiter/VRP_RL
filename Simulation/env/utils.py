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

def flatten_dict(d):
    ret = {}
    for k in d:
        if isinstance(d[k], dict):
            d2 = flatten_dict(d[k])
            for k2 in d2:
                new_key = "{}_{}".format(k, k2)
                ret[new_key] = d2[k2]
        else:
            ret[k] = d[k]
    return ret