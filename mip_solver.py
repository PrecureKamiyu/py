import numpy as np
import pandas as pd
from mip import Model, xsum, minimize, BINARY
from placement_metrics import calculate_access_delay, calculate_workload_balance


df = pd.read_csv('./shanghai_dataset/block_counts.csv')
lat = np.array(df['lat_block'])
lon = np.array(df['lon_block'])
points = np.column_stack((lat, lon))
blocks = np.column_stack((lat, lon))

def mip_solver(
        number_of_variables=20,
        number_of_server=5,
        weight=0.5,
        balances_path='balances.npy',
        distances_path='distances.npy',
        locations_path='locations_to_be_chosen.npy'
):
    balances = np.load(balances_path)
    distances= np.load(distances_path)
    possible_locations = np.load(locations_path)
    possible_locations = possible_locations.tolist()

    m = Model("test1")
    x = [m.add_var(var_type=BINARY) for _ in range(number_of_variables)]
    m.objective = \
        minimize(xsum(weight * x[i] * balances[i] + (1-weight) * x[i] * distances[i] \
                      for i in range(number_of_variables)))
    m += xsum(x[i] for i in range(number_of_variables)) >= number_of_server
    m += xsum(x[i] for i in range(number_of_variables)) >= number_of_server
    m.optimize()

    selected = [i for i in range(number_of_variables) if x[i].x >= 0.99]
    selected_locations = [
        possible_locations[i] for i in selected
    ]
    print(selected)
    return {
        'weight': weight,
        'selected': selected,
        'placement_location': selected_locations,
        'balance':  sum([balances[i] for i in selected]),
        'distance': sum([distances[i] for i in selected]),
        # 'balance':  calculate_workload_balance(selected_locations, df),
        # 'distance': calculate_access_delay(selected_locations, df)
    }


if __name__ == "__main__":
    print(mip_solver())
