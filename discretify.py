import pandas as pd
import numpy as np
import math


df = pd.read_csv('./shanghai_dataset/block_counts.csv')
possible_locations = np.load('locations_to_be_chosen.npy')
number_of_possible_locations = 20
number_of_chosed_locations = 5
number_of_cluster = 4

lat = np.array(df['lat_block'])
lon = np.array(df['lon_block'])
points = np.column_stack((lat, lon))
blocks = np.column_stack((lat, lon))
number_of_blocks = len(points)

counts = df['count']
sum_counts = counts.sum()


distances = np.zeros(number_of_possible_locations)

for i in range(len(possible_locations)):
    distances_location_i_to_other_location = \
        [(math.dist(possible_locations[i], location), j) for j, location in enumerate(possible_locations)]
    sorted_distances = sorted(distances_location_i_to_other_location)
    # Kth_nearest_distance_tuple = sorted_distances[:number_of_cluster+1][-1]
    # print(Kth_nearest_distance_tuple)
    distances[i] = sorted_distances[:number_of_cluster+1][-1][0]

distances = distances / number_of_chosed_locations

balances = np.zeros(number_of_possible_locations)

for i in range(number_of_blocks):
    distances_block_i_to_locations = \
        [(math.dist(blocks[i], location), j) for j, location in enumerate(possible_locations)]
    sorted_distances = sorted(distances_block_i_to_locations)
    closest_location_index = sorted_distances[0][1]
    balances[closest_location_index] += counts[i]

mean_balance = np.mean(balances)
balances = balances / mean_balance / number_of_chosed_locations

np.save('distances.npy', distances)
np.save('balances.npy', balances)
