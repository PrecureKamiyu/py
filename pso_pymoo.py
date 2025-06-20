# https://pymoo.org/algorithms/soo/pso.html
import pandas as pd
import numpy as np

from placement_metrics import calculate_access_delay, calculate_workload_balance

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.problems.single import Rastrigin
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize

from problem_in_pymoo import OneProblem


df = \
    pd.read_csv('./shanghai_dataset/block_counts.csv')

lst = [0.2, 0.3, 0.5, 0.7, 0.8]

points = []

for idx, alpha in enumerate(lst):
    print("========")
    print(f"{idx}-th round iteration begins, alpha is {alpha}")
    problem = OneProblem(alpha)
    algorithm = PSO()

    res = minimize(problem,
                   algorithm,
                   seed=1,
                   verbose=True)

    # res.F would a list that has one element
    # points = points.append([res.F[0] / alpha, res.F[0] / (1 - alpha)])

    points.append([
        calculate_workload_balance(np.array(res.X).reshape(-1,2), df),
        calculate_access_delay(np.array(res.X).reshape(-1,2), df)
    ])


def helper(F):
    import numpy as np
    farr = np.array(F)
    np.save('pso_pymoo_fronts.npy', farr)
    np.savetxt('my_array_from_pso.txt', farr, delimiter=',')

helper(points)

plot = Scatter()
plot.add(points, facecolor="none", edgecolor="red")
plot.show()
