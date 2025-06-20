from scipy.optimize import milp, Bounds, LinearConstraint
import numpy as np
import pandas as pd

# 1. Define the objective function coefficients (c)
c = np.array([-1, -4])  # Minimize -x - 4y

# 2. Define the bounds for the variables
bounds = Bounds(0, None)  # x >= 0, y >= 0

# 3. Define the linear constraints
linear_constraints = [
    LinearConstraint([-3, 1], -np.inf, 6),  # -3x + y <= 6
    LinearConstraint([1, 2], -np.inf, 4)   # x + 2y <= 4
]

# 4. Specify the integrality of the variables
integrality = np.array([0, 1])  # x is continuous (0), y is integer (1)

# 5. Solve the MIP
result = milp(c, constraints=linear_constraints, bounds=bounds, integrality=integrality)

# 6. Prepare data for CSV
if result.success:
    data = {
        'Variable': ['x', 'y', 'Objective Value'],
        'Value': [result.x[0], result.x[1], result.fun]
    }
    df = pd.DataFrame(data)

    # 7. Save to CSV
    csv_filename = 'milp_results.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

    print("\nOptimal solution found:")
    print(f"x = {result.x[0]:.4f}")
    print(f"y = {result.x[1]:.4f}")
    print(f"Optimal objective value = {result.fun:.4f}")
else:
    print("No optimal solution found.")
