import numpy as np
from PFL_v1 import plot_pareto_front_and_save

def is_dominated(a, b, minimize=True):
    # Returns True if 'a' is dominated by 'b'
    if minimize:
        return all(b_i <= a_i for b_i, a_i in zip(b, a)) and any(b_i < a_i for b_i, a_i in zip(b, a))
    else:
        return all(b_i >= a_i for b_i, a_i in zip(b, a)) and any(b_i > a_i for b_i, a_i in zip(b, a))


def get_pareto_front(points, minimize=True):
    pareto_front = []
    for i, a in enumerate(points):
        dominated = False
        for j, b in enumerate(points):
            if i != j and is_dominated(a, b, minimize):
                dominated = True
                break
        if not dominated:
            pareto_front.append(a)
    return pareto_front


for i in range(6):
    points = np.load(f"points3_{i}.npy")
    pareto_front = get_pareto_front(points, minimize=True)
    plot_pareto_front_and_save(pareto_front, f"./fig/test_{i}.png")
