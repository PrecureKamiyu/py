import numpy as np

def calculate_hypervolume(pareto_front, offset=1.0):
    """
    Calculate the Hypervolume (HV) of a 2D Pareto front with a default reference point.

    Parameters:
    - pareto_front: 2D array-like, e.g., [[x1, y1], [x2, y2], ...]
    - offset: Positive value added to max(x) and max(y) for the reference point (default: 1.0)

    Returns:
    - hv: Hypervolume (area dominated by the Pareto front)
    """
    # Convert to NumPy array
    front = np.array(pareto_front)

    # Sort by x-coordinate
    front = front[front[:, 0].argsort()]

    # Extract x and y
    x = front[:, 0]
    y = front[:, 1]

    # Default reference point: max values + offset
    ref_x = np.max(x) + offset
    ref_y = np.max(y) + offset
    reference_point = [ref_x, ref_y]

    # Add boundary points
    x_full = np.concatenate(([x[0]], x, [ref_x]))
    y_full = np.concatenate(([ref_y], y, [y[-1]]))

    # Calculate area
    hv = 0.0
    for i in range(len(x_full) - 1):
        width = x_full[i + 1] - x_full[i]
        height = ref_y - y_full[i]
        hv += width * height

    return hv

if __name__ == "main":
    # Example usage
    pareto_front = [[1, 4], [2, 3], [4, 1]]
    hv = calculate_hypervolume(pareto_front, offset=1.0)
    print(f"Hypervolume: {hv}")  # Uses [5, 5] as reference point
