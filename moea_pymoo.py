from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

from problem_in_pymoo import PlacementProblemN, PlacementProblemNprime


def main(
        number_of_servers=20,
        number_of_moea_gen=20,
        path_to_saved_result="moea_pymoo.npy",
        version_string="foo",
        verbose=False,
):
    print("Number of servers", number_of_servers)
    print("Number of generation", number_of_moea_gen)
    print("Path to saved result:", path_to_saved_result)
    problem = PlacementProblemNprime(number_of_servers)
    ref_directions = get_reference_directions("uniform", 2, n_partitions=12)
    algorithm = MOEAD(
        ref_directions,
        n_neighbors=15,
        prob_neighbor_mating=0.7,
    )
    res = minimize(problem,
                   algorithm,
                   ('n_gen', number_of_moea_gen),
                   seed=1,
                   verbose=True)
    import numpy as np
    res_array = np.array(res.F)
    path = path_to_saved_result
    np.save(path, res_array)
    np.save(f"moea_pymoo_{number_of_servers}_servers", res_array)
    np.save(f"moea_pymoo_{number_of_servers}_servers_{version_string}", res_array)
    np.savetxt('moea_pymoo.txt', res_array, delimiter=',')
    if verbose:
        Scatter().add(res.F).show()
    return {
        "placement_location": (res.X).tolist(),
        "values": (res.F).tolist(),
    }


if __name__ == "__main__":
    main(verbose=True)
