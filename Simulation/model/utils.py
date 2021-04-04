def powerset(s):
    res = []
    x = len(s)
    for i in range(1 << x):
        res.append([s[j] for j in range(x) if (i & (1 << j))])
    return res