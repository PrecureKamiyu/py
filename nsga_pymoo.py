# https://pymoo.org/algorithms/moo/nsga2.html
#
# when you turn on the verbose option for the algorithm
# you can see the indicator along
# which can be used
# https://stackoverflow.com/questions/77390883/pymoo-nsga2-how-to-interpret-in-output-indicator-column-f-indicator
#
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from problem_in_pymoo import PlacementProblemN
# from pymoo.visualization.scatter import Scatter

def main(
        number_of_servers=20,
        number_of_generation=40,
        path_to_saved_result="nsga_fronts.npy",
):
    print("Number of servers", number_of_servers)
    print("Number of generation", number_of_generation)
    print("Path to saved result:", path_to_saved_result)
    problem = PlacementProblemN(number_of_servers)
    algorithm = NSGA2(pop_size=200)
    res = minimize(problem,         #
                   algorithm,       #
                   ('n_gen', number_of_generation),
                   seed=1,          #
                   verbose=True)    #
    F = res.F
    farr = np.array(F)
    np.save(path_to_saved_result, farr)
    np.save(f"nsga_fronts_{number_of_servers}_servers.npy", farr)
    np.savetxt('my_array.txt', farr, delimiter=',')
    return {
        "placement_location": (res.X).tolist(),
        "values": (res.F).tolist(),
    }


# plot = Scatter()
# plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res.F, facecolor="none", edgecolor="red")
# print(res.F)
