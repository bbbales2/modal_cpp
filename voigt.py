#%%
voigt = [[(0, 0)], [(1, 1)], [(2, 2)], [(1, 2), (2, 1)], [(0, 2), (2, 0)], [(0, 1), (1, 0)]]

for i in range(6):
    for j in range(6):
        for k, l in voigt[i]:
            for n, m in voigt[j]:
                print "C({k} + {l} * 3, {n} + {m} * 3) = Ch({i}, {j});".format(i = i, j = j, k = k, l = l, n = n, m = m)
