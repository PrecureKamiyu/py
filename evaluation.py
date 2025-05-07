import numpy as np
from scipy.spatial.distance import cdist

def pareto_front_similarity(A, B):
    """
    Compute the Hausdorff distance between two Pareto fronts after normalizing objectives.

    Parameters:
    A (numpy.ndarray): Array of shape (n_points_A, n_objectives) for the first Pareto front.
    B (numpy.ndarray): Array of shape (n_points_B, n_objectives) for the second Pareto front.

    Returns:
    float: Hausdorff distance between the normalized Pareto fronts (smaller = more similar).
    """
    # Ensure both fronts have the same number of objectives
    assert A.shape[1] == B.shape[1], "Number of objectives must be the same"

    # Combine points to find global min and max for each objective
    all_points = np.vstack((A, B))
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)

    # Compute the range for each objective, avoiding division by zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # If range is 0, normalized value becomes 0

    # Normalize both fronts to [0,1]
    normalized_A = (A - min_vals) / range_vals
    normalized_B = (B - min_vals) / range_vals

    # Compute pairwise Euclidean distances between points
    D = cdist(normalized_A, normalized_B)

    # Minimum distance from each point in A to nearest in B
    min_dists_A_to_B = np.min(D, axis=1)
    max_min_A_to_B = np.max(min_dists_A_to_B)

    # Minimum distance from each point in B to nearest in A
    min_dists_B_to_A = np.min(D, axis=0)
    max_min_B_to_A = np.max(min_dists_B_to_A)

    # Hausdorff distance is the maximum of the two directed distances
    hausdorff_distance = max(max_min_A_to_B, max_min_B_to_A)

    return hausdorff_distance

# Example usage
if __name__ == "__main__":
    # Sample Pareto fronts (2 objectives: minimizing both)
    A = np.array([[0, 1], [0.5, 0.5], [1, 0]])      # Front 1
    B = np.array([[0.1, 0.9], [0.6, 0.4], [1.1, 0]]) # Front 2

    distance = pareto_front_similarity(A, B)
    print(f"Hausdorff Distance: {distance:.4f}")
