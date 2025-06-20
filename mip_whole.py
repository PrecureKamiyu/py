# * Parameter and setting

n = 10
N = 80
option_if_regenerate_locations = True
# option_if_regenerate_locations = False
number_of_servers = n
experiment_name = f"_testing_n{n}_N{N}"


print("Startup")
print("- Number of servers", n)

# * locations generation

import mip_location_generate
if option_if_regenerate_locations:
    mip_location_generate.generate_to_be_chosen_locations(
        number_of_to_be_chosed_locations=N,
        number_of_set=3
    )

# * draw locations distribution (optional)

# import mip_draw_generated_locations
# mip_draw_generated_locations.draw_locations_to_be_chosen(
#     index=2
# )

# * discretify to make it possible to be solved

import mip_discretify
for i in range(1,4):
    mip_discretify.mip_discretify(
        number_of_server=n,
        path_to_possible_locations=f'locations_to_be_chosen_{i}.npy',
        path_to_balances=f"balances_{i}.npy",
        path_to_distances=f"distances_{i}.npy"
    )


# * solve and store some values

import mip_solver
import numpy as np
weights = [i / 40.0 for i in range(10,25)]

# helper will collect the front
# helper will collect the placement and something
# i will range from 0 to 2
# which is the id for the possible locations (potential locations)
def helper(i):
    points = []
    record = {
        'id': i,
        'weights': weights,
        'records': [],
    }
    for weight in weights:
        ret = mip_solver.mip_solver(number_of_variables=N,
                                    number_of_server=n,
                                    weight=weight,
                                    balances_path=f'balances_{i}.npy',
                                    distances_path=f'distances_{i}.npy',
                                    locations_path=f'locations_to_be_chosen_{i}.npy')
        points.append([ret['balance'], ret['distance']])
        record['records'].append(ret)
    return (np.array(points), record)

list_of_fronts_and_records = [helper(i) for i in range(1,4)]
list_of_pareto_fronts, list_of_records = map(list, zip(*list_of_fronts_and_records))

for i, front in enumerate(list_of_pareto_fronts):
    np.save(f"mip_front_{i+1}", front)
    np.save(f"mip_front_{i+1}_{number_of_servers}_servers", front)
    print("save to", f"mip_front_{i+1}")
    print("save to", f"mip_front_{i+1}_{number_of_servers}_servers")

# * try to store the points

def show(index,path=''):
    import matplotlib.pyplot as plt
    front = list_of_pareto_fronts[index]
    xs = [point[0] for point in front]
    ys = [point[1] for point in front]
    plt.scatter(xs, ys)
    plt.grid(True)
    if path != '':
        plt.savefig(path)
    else:
        plt.show()
    plt.close()


for i in range(1,4):
    show(i-1, f"./fig/mip_test_{i}.png")

# * save the result as json

import json_numpy as json
data = {
    'number_of_servers': n,
    'number_of_potential_servers': N,
    'records': list_of_records,
}
json_file_path = f"./records/mip{experiment_name}.json"
with open(json_file_path, 'w') as json_file:
    json.dump(data, json_file, indent=2)
print("record stores to", json_file_path)
