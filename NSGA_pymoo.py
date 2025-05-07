# Install pymoo if not already installed
# !pip install pymoo

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling  # Updated sampling
from pymoo.operators.crossover.sbx import SBX  # Updated crossover
from pymoo.operators.mutation.pm import PM  # Updated mutation
from pymoo.optimize import minimize
import matplotlib.pyplot as plt


# Step 1: Define the MOO problem
class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=30,  # Number of decision variables
                         n_obj=2,   # Number of objectives
                         n_constr=0,  # Number of constraints
                         xl=0,       # Lower bound for variables
                         xu=1)       # Upper bound for variables

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = X[:, 0]  # First objective
        g = 1 + 9 / (self.n_var - 1) * np.sum(X[:, 1:], axis=1)
        f2 = g * (1 - np.sqrt(f1 / g))  # Second objective

        out["F"] = np.column_stack([f1, f2])


# Step 2: Set up and run NSGA-II
problem = MyProblem()

algorithm = NSGA2(
    pop_size=100,  # Population size
    sampling=FloatRandomSampling(),  # Updated sampling method
    crossover=SBX(prob=0.9, eta=15),  # Updated crossover operator
    mutation=PM(eta=20),  # Updated mutation operator
    eliminate_duplicates=True
)

res = minimize(problem,
               algorithm,
               ('n_gen', 250),  # Number of generations
               seed=1,
               verbose=True)

# Step 3: Plot the Pareto front
F = res.F  # Extract the Pareto front

plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Pareto Front")
plt.xlabel("Objective 1")
plt.ylabel("Objective 2")
plt.show()
