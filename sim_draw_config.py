import json
import matplotlib.pyplot as plt
from sim_generate_json import index_to_location

def main(path_to_config="test_config.json"):
    data = None
    with open(path_to_config, 'r') as json_file:
        data = json.load(json_file)
    ns_locations = \
        [ns["attributes"]["coordinates"] for ns in data["NetworkSwitch"]]
    xs, ys = zip(*ns_locations)
    plt.scatter(xs, ys, c='blue')

    nl_locations = \
        [list(map(index_to_location, (nl["relationships"]["nodes"][0]["id"], nl["relationships"]["nodes"][1]["id"]))) for nl in data["NetworkLink"]]
    nl_nodes = \
        [(nl["relationships"]["nodes"][0]["id"], nl["relationships"]["nodes"][1]["id"]) for nl in data["NetworkLink"]]
    # print(nl_locations)
    print(nl_nodes)
    for start, end in nl_locations:
        plt.plot([start[0], end[0]], [start[1], end[1]], c='green')


    service_server_id = \
        [service["relationships"]["server"]["id"] for service in data["Service"]]
    servers = data["EdgeServer"]
    service_locations = [servers[idx]["attributes"]["coordinates"] for idx in service_server_id]
    xs, ys = zip(*service_locations)
    plt.scatter(xs, ys, c='red')

    plt.plot(True)
    plt.show()


main()
