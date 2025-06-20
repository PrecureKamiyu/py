import json as js
def main_one():
    path = "./records/mip_testing_n20_N80.json"


    data = None
    with open(path, 'r') as js_file:
        data = js.load(js_file)

    one_exp = data['records'][0]
    target_weight = 0.5
    weights = one_exp['weights']

    # print(all(weight == target_weight for weight in weights))
    # print(weights.index(target_weight))

    index = weights.index(target_weight)
    locations = one_exp['records'][index]['placement_location']
    print(locations)
    return locations


def main():
    path = "./records/moea_10_servers.json"
    data = None
    with open(path, 'r') as js_file:
        data = js.load(js_file)

    target_weight = 0.5
    values = data['values']
    locationss = data['placement_location']

    weighted_values = \
        [target_weight * f1 + (1-target_weight) * f2 for f1, f2 in values]
    index_for_lowest_weighted_value = \
        weighted_values.index(min(weighted_values))
    locations = locationss[index_for_lowest_weighted_value]
    return list(zip(locations[::2], locations[1::2]))


locations = main()
print(locations)
print(len(locations))
