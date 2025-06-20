import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import find_weighted_solution

ns = [5,10,20,30]
# objective = "balance"
objective = "delay"


paths_to_moea = [f"moea_pymoo_{n}_servers.npy" for n in ns]
moea_balances = [find_weighted_solution.main(path)["balance"] for path in paths_to_moea]
moea_delays   = [find_weighted_solution.main(path)["delay"]   for path in paths_to_moea]


data = {
    'Proposed Method': [find_weighted_solution.main(path)[objective] for path in paths_to_moea],
}

for i in range(1,4):
    paths_to_mip = [f"mip_front_{i}_{n}_servers.npy" for n in ns]
    mip_balances = [find_weighted_solution.main(path)["balance"] for path in paths_to_mip]
    mip_delays   = [find_weighted_solution.main(path)["delay"] for path in paths_to_mip]

    data[f"cMIP_L{i}"] = [find_weighted_solution.main(path)[objective] for path in paths_to_mip]
    # data[f"mip_delays_{i}"]   = mip_delays

df_wide = pd.DataFrame(data, index=['5', '10', '20', '30'])
ax = df_wide.plot(kind='bar', figsize=(10, 6), rot=0,
                  title='Comparison')
ax.set_xlabel('Number of server')
ax.set_ylabel(objective)

plt.grid(axis='y', linestyle='--', alpha=0.7) # Optional: add a grid
plt.tight_layout()
plt.show()
