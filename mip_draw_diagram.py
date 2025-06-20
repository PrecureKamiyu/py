import matplotlib.pyplot as plt
import find_weighted_solution


def main(
        weight=0.5,
):

    path_to_mip_front = "mip_front_1.npy"
    path_to_moea_front = "moea_pymoo.npy"
    res_mip = find_weighted_solution.main(path=path_to_mip_front,
                                          weight=weight)
    res_moea = find_weighted_solution.main(path=path_to_moea_front,
                                           weight=weight)
    
