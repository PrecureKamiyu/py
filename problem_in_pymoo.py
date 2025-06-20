# https://pymoo.org/problems/definition.html

import numpy as np
import pandas as pd
from placement_metrics import calculate_access_delay, calculate_workload_balance
from pymoo.core.problem import Problem

df = \
    pd.read_csv('./shanghai_dataset/block_counts.csv')


class TestProblem(Problem):

    def __init__(self):
        super().__init__(n_var=4,
                         n_obj=2,
                         n_ieq_constr=0,
                         xl=0.0,
                         xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        # xreshape = np.array(x).reshape(-1,2)
        fst = [calculate_workload_balance(np.array(y).reshape(-1,2), df) for y in x]
        snd = [calculate_access_delay(np.array(y).reshape(-1,2), df) for y in x]
        out["F"] = [fst, snd]


class PlacementProblem2(Problem):

    def __init__(self):
        super().__init__(n_var=4,
                         n_obj=2,
                         n_ieq_constr=0,
                         xl=0.0,
                         xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        fst = [calculate_workload_balance(np.array(y).reshape(-1,2), df) for y in x]
        snd = [calculate_access_delay(np.array(y).reshape(-1,2), df) for y in x]
        out["F"] = [fst, snd]


class PlacementProblem3(Problem):

    def __init__(self):
        super().__init__(n_var=6,
                         n_obj=2,
                         n_ieq_constr=0,
                         xl=0.0,
                         xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        fst = [calculate_workload_balance(np.array(y).reshape(-1,2), df) for y in x]
        snd = [calculate_access_delay(np.array(y).reshape(-1,2), df) for y in x]
        out["F"] = [fst, snd]


class PlacementProblem10(Problem):

    def __init__(self):
        super().__init__(n_var=20,
                         n_obj=2,
                         n_ieq_constr=0,
                         xl=0.0,
                         xu=1.0)


    def _evaluate(self, x, out, *args, **kwargs):
        fst = [calculate_workload_balance(np.array(y).reshape(-1,2), df) for y in x]
        snd = [calculate_access_delay(np.array(y).reshape(-1,2), df) for y in x]
        out["F"] = [fst, snd]


class PlacementProblemN(Problem):

    def __init__(self, number_of_servers=5):
        super().__init__(n_var=2 * number_of_servers,
                         n_obj=2,
                         n_ieq_constr=0,
                         xl=0.0,
                         xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        fst = [calculate_workload_balance(np.array(y).reshape(-1,2), df) for y in x]
        snd = [calculate_access_delay(np.array(y).reshape(-1,2), df) for y in x]
        out["F"] = [fst, snd]



class OneProblem(Problem):

    def __init__(self, alpha):
        super().__init__(n_var=4,
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=0.0,
                         xu=1.0)
        self.alpha = alpha

    def _evaluate(self, x, out, *args, **kwargs):
        # xreshape = np.array(x).reshape(-1,2)
        fst = [calculate_workload_balance(np.array(y).reshape(-1,2), df) for y in x]
        snd = [calculate_access_delay(np.array(y).reshape(-1,2), df) for y in x]


        ans = [self.alpha * x + (1 - self.alpha) * y for x, y in zip(fst, snd)]
        out["F"] = [ans]

# print(f"G: {G}\n")

if __name__=="__main__":
    problem = TestProblem()
    F = problem.evaluate(np.random.rand(3, 4))

    print(f"F: {F}\n")
