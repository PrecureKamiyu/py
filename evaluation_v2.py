# Evaluation implement the IGD and hausdorff distance.
# I mean the they are actually very simple.
# \f[H(A, B) = \max \left( \max_{a \in A} \min_{b \in B} d(a, b), \max_{b \in B} \min_{a \in A} d(a, b) \right)\f]
#
# \f[IGD(A, B) = \frac{1}{|A|} \sum_{a \in A} \min_{b \in B} d(a, b)\f]

# find the reference for the similarity
# 

import numpy as np
from scipy.spatial.distance import cdist

def find_overlapping_region(front1, front2):
    min_x = max(np.min(front1[:, 0]), np.min(front2[:, 0]))
    max_x = min(np.max(front1[:, 0]), np.max(front2[:, 0]))
    min_y = max(np.min(front1[:, 1]), np.min(front2[:, 1]))
    max_y = min(np.max(front1[:, 1]), np.max(front2[:, 1]))
    return (min_x, max_x), (min_y, max_y)

def extract_overlapping_points(front, x_range, y_range):
    x_min, x_max = x_range
    y_min, y_max = y_range
    mask = (front[:, 0] >= x_min) & (front[:, 0] <= x_max) & \
           (front[:, 1] >= y_min) & (front[:, 1] <= y_max)
    return front[mask]

def normalize_front(pareto_front):
    min_vals = np.min(pareto_front, axis=0)
    max_vals = np.max(pareto_front, axis=0)
    return (pareto_front - min_vals) / (max_vals - min_vals)

def hausdorff_distance(front1, front2):
    dist_matrix = cdist(front1, front2, metric='euclidean')
    return max(np.max(np.min(dist_matrix, axis=1)), np.max(np.min(dist_matrix, axis=0)))

def igd(reference_front, front):
    dist_matrix = cdist(reference_front, front, metric='euclidean')
    return np.mean(np.min(dist_matrix, axis=1))

def compare_pareto_fronts(front1, front2):
    """Return a dictionary of two member"""
    # Step 1: Find the overlapping region
    x_range, y_range = find_overlapping_region(front1, front2)

    # Step 2: Extract points within the overlapping region
    overlapping_front1 = extract_overlapping_points(front1, x_range, y_range)
    overlapping_front2 = extract_overlapping_points(front2, x_range, y_range)

    # Step 3: Normalize the overlapping points
    normalized_front1 = normalize_front(overlapping_front1)
    normalized_front2 = normalize_front(overlapping_front2)

    # Step 4: Compute similarity metrics
    if len(normalized_front1) > 0 and len(normalized_front2) > 0:
        hausdorff_dist = hausdorff_distance(normalized_front1, normalized_front2)
        igd_value = igd(normalized_front1, normalized_front2)
        return {
            "hausdorff_distance": hausdorff_dist,
            "igd": igd_value
        }
    else:
        return None
