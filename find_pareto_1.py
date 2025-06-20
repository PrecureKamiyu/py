# * find pareto front
from paretoset import paretoset
import numpy as np
import pandas as pd

path = "moea_pymoo.npy"
# path = "nsga_pymoo_fronts.npy"
locations = np.load(path)

df = pd.DataFrame({
    'f1': [location[0] for location in locations],
    'f2': [location[1] for location in locations]
})
mask = paretoset(df, sense=["min", "min"])
paretoset_data = df[mask]
print(paretoset_data)

# * draw the front
import matplotlib.pyplot as plt
plt.scatter(np.array(paretoset_data['f1']), np.array(paretoset_data['f2']))
plt.title(path)
plt.grid(True) # Optional: Add a grid
plt.show()

# * draw the front (not strictly front)
import matplotlib.pyplot as plt
plt.scatter([location[0] for location in locations],
            [location[1] for location in locations])
plt.title(path)
plt.grid(True)
plt.show()

# * find the hv

from pymoo.indicators.hv import HV
ref_point = np.array([1.1, 1.1])
ind = HV(ref_point=ref_point)
print(path, "HV", ind(locations))
