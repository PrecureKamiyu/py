from mip import Model, xsum, maximize, BINARY

p = [10, 13, 18, 31, 7, 15]
w = [11, 15, 20, 35, 10, 33]
c, I = 47, range(len(w))

m = Model("knapsack")


# 待定的 N = 2 * n 就行了
# 目标计算的数据集 df
# 真的放置的数量 n
x = [m.add_var(var_type=BINARY) for i in I]

# 目标
m.objective = maximize(xsum(p[i] * x[i] for i in I))

# 限制条件的
m += xsum(w[i] * x[i] for i in I) <= c

m.optimize()

selected = [i for i in I if x[i].x >= 0.99]
print("selected items:{}".format(selected))
