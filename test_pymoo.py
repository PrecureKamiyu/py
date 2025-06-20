# test for the concern of the calculation of HV


from pymoo.indicators.hv import HV
import numpy as np

# Example: Front A (10 points) and Front B (20 points), 2 objectives, minimization
front_a = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1], ...])  # 10 points
front_b = np.array([[1.1, 5.2], [2.1, 4.1], ...])  # 20 points

# Reference point (worse than all points, e.g., slightly beyond max values)
ref_point = np.array([10, 10])

# Compute hypervolume
hv = HV(ref_point=ref_point)
hv_a = hv(front_a)
hv_b = hv(front_b)

# Subsample Front B to 10 points, repeat N times
N = 100
hv_b_subsampled = []
for _ in range(N):
    indices = np.random.choice(len(front_b), size=10, replace=False)
    hv_b_subsampled.append(hv(front_b[indices]))
hv_b_avg = np.mean(hv_b_subsampled)

# Compare
print(f"Hypervolume Front A: {hv_a}")
print(f"Average Hypervolume Front B (subsampled): {hv_b_avg}")
