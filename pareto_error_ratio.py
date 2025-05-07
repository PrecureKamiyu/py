import numpy as np

def calculate_pcer(approx_front, true_front):
    """
    Calculate the Pareto Coverage Error Ratio (PCER) of an approximated 2D Pareto front
    compared to the true front. Assumes minimization objectives.

    Parameters:
    - approx_front: 2D array-like, e.g., [[x1, y1], [x2, y2], ...], approximated front
    - true_front: 2D array-like, e.g., [[x1, y1], [x2, y2], ...], true front

    Returns:
    - pcer: Fraction of approx_front points dominated by at least one true_front point
    """
    # Convert to NumPy arrays
    A = np.array(approx_front)
    T = np.array(true_front)

    if A.shape[1] != 2 or T.shape[1] != 2:
        raise ValueError("Both fronts must be 2D arrays with shape (n, 2).")

    n = len(A)
    if n == 0:
        return 0.0  # No points, no error

    # Count dominated points
    n_d = 0
    for a in A:
        # Check if any point in T dominates a
        dominated = np.any(
            (T[:, 0] <= a[0]) & (T[:, 1] <= a[1]) &  # T is better or equal in both
            ((T[:, 0] < a[0]) | (T[:, 1] < a[1]))    # T is strictly better in at least one
        )
        if dominated:
            n_d += 1

    # Pareto Coverage Error Ratio
    pcer = n_d / n
    return pcer


if __name__ == "main":
    approx_front = [[1, 4], [2, 3], [4, 1], [3, 5]]  # Approximated front
    true_front = [[1, 4], [2, 3], [4, 1]]            # True front
    pcer = calculate_pcer(approx_front, true_front)
    print(f"Pareto Coverage Error Ratio: {pcer}")  # Should be 0.25
