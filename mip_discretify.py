import pandas as pd
import numpy as np
import math



def mip_discretify(
        number_of_server=5,
        path_to_possible_locations='locations_to_be_chosen.npy',
        path_to_balances='balances.npy',
        path_to_distances='distances.npy'):
    """
    函数：将优化目标离散化
    """
    possible_locations = np.load(path_to_possible_locations,)
    number_of_possible_locations = len(possible_locations)
    number_of_cluster = round(number_of_possible_locations / number_of_server) # cluster size

    df = pd.read_csv('./shanghai_dataset/block_counts.csv')
    lat = np.array(df['lat_block'])
    lon = np.array(df['lon_block'])
    points = np.column_stack((lat, lon))
    blocks = np.column_stack((lat, lon))
    number_of_blocks = len(points)

    counts = df['count']
    block_counts = counts
    sum_counts = counts.sum()

    # Find the neighbor_indice
    # which is i-th location's K nearest locations' indice
    neighbor_locations_indices = [[] for _ in range(number_of_possible_locations)]
    for i in range(number_of_possible_locations):
        distances_location_i_to_other_location = \
            [(math.dist(possible_locations[i], location), j) for j, location in enumerate(possible_locations)]
        sorted_distances = sorted(distances_location_i_to_other_location)
        K_nearest_possible_location_indice = [t[1] for t in sorted_distances[:number_of_cluster+1]]
        neighbor_locations_indices[i] = K_nearest_possible_location_indice

    distances = np.zeros(number_of_possible_locations)
    for b_idx in range(number_of_blocks):
        distances_b_to_locations = \
            [(math.dist(blocks[b_idx], location), j) for j, location in enumerate(possible_locations)]
        distances_b_to_locations = sorted(distances_b_to_locations)
        for i in range(number_of_possible_locations):
            distance_to_nearest_location, nearest_location_index = \
                distances_b_to_locations[0]
            if nearest_location_index in neighbor_locations_indices[i]:
                distances[i] += distance_to_nearest_location * counts[b_idx]
    for i in range(len(distances)):
        if distances[i] < 0.1:
            distances[i] += 100
    distances = 20 * distances / sum_counts # 20 is a magic number

    # NOTE: this is the old discretification of distances
    #       this is achieved by calculating the distances between the locations
    #       rather than the user to servers
    # distances = np.zeros(number_of_possible_locations)
    # for i in range(len(possible_locations)):
    #     distances_location_i_to_other_location = \
    #         [(math.dist(possible_locations[i], location), j) for j, location in enumerate(possible_locations)]
    #     sorted_distances = sorted(distances_location_i_to_other_location)
    #     distances[i] = sorted_distances[:number_of_cluster][-1][0]
    #     # distances[i] = sorted_distances[:number_of_cluster][-1][0] / 2
    #     # distances[i] = np.mean([d_tuple[0] for d_tuple in sorted_distances[:round(number_of_cluster / 2)]])
    # distances = distances / number_of_server # NOTE: this is added afterward

    workloads = np.zeros(number_of_possible_locations)

    for i in range(number_of_blocks):
        distances_block_i_to_locations = \
            [(math.dist(blocks[i], location), j) for j, location in enumerate(possible_locations)]
        sorted_distances = sorted(distances_block_i_to_locations)
        closest_location_index = sorted_distances[0][1]
        workloads[closest_location_index] += counts[i]

    mean_workload = np.mean(workloads)
    workloads = workloads / mean_workload
    balances = np.array([abs(w - 1) for w in workloads])
    balances = balances / number_of_server

    np.save(path_to_distances, distances)
    np.save(path_to_balances, balances)

if __name__ == "__main__":
    mip_discretify()
