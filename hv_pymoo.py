import numpy as np
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter
from pymoo.indicators.hv import HV

# ref_point = np.array([1.2, 1.2])

# ind = HV(ref_point=ref_point)
# print("HV", ind(A))

# # The pareto front of a scaled zdt1 problem
# pf = get_problem("zdt1").pareto_front()

# # The result found by an algorithm
# A = pf[::10] * 1.1

data = np.load('nsga_pymoo_fronts.npy')
ref_point = np.array([
    np.max(data[:, 0]),
    np.max(data[:, 1])
])
ref_point = np.array([0.8, 0.3])
ind = HV(ref_point=ref_point)
print("HV", ind(data))
