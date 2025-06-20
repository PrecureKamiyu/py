import json
import math
import random

from edge_sim_py.components import network


def singleNS(
        _id=1,
        coordinates=[0,0],
        active=True
):
    attributes = {
        "id": _id,
        "coordinates": coordinates,
        "active": active,
        "power_model_parameters": {
            "chassis_power": 60,
            "ports_power_consumption": {
                "125": 1,
                "12.5": 0.3
            }
        }
    }

    return {
        'attributes': attributes,
        'relationships': {},
    }

def singleNL(
        _id=1,
        delay=5,
        bandwidth=12.5,
        bandwidth_demand=0,
        active=True
):
    attributes = {
        "id": _id,
        "delay": delay,
        "bandwidth": bandwidth,
        "bandwidth_demand": bandwidth_demand,
        "active": active
    }
    relationships = {
        "topology": {
            "class": "Topology",
            "id": 1,
        },
        "active_flows": [],
        "applications": [],
        "nodes": [
            {
                "class": "NetworkSwitch",
                "id": 1
            },
            {
                "class": "NetworkSwitch",
                "id": 5,
            }
        ]
    }

    return {
        "attributes": attributes,
        "relationships": relationships
    }

def singleBS(
        _id=1,
        coordinates=[0,0],
        wireless_delay=5
):
    attributes = {
        "id": _id,
        "coordinates": coordinates,
        "wireless_delay": wireless_delay
    }
    relationships = {}
    return {
        "attributes": attributes,
        "relationships": relationships
    }


def singleUser(
        _id=1,
        coordinates=[0,0],
):
    attributes = {
        "id": _id,
        "delays": {
            f"{_id}": 10
        },
        "delay_slas": {
            f"{_id}": 45
        },
        "communication_paths": {
            # 每个 user 需要的 application 就有一个
            f"{_id}": []
        },
        "making_requests": {
            # this is the id of application
            f"{_id}": {
                "1": True
            }
        },
        "coordinates": coordinates,
        "coordinates_trace ": [coordinates for _ in range(10)]
    }

    relationships = {}
    return {
        "attributes": attributes,
        "relationships": relationships
    }

def singleContainerLayer(
        _id=1,
        size=2,
        digest="sha256:df9b9388f04ad6279a7410b85cedfdcb2208c0a003da7ab5613af71079148139",
        instruction="ADD file:5d673d25da3a14ce1f6cf"
):
    attributes = {
        "id": _id,
        "digest": digest,
        "size": size,
        "instruction": instruction
    }
    relationships = {}
    return {
        "attributes": attributes,
        "relationships": relationships
    }

def singleContainerImage(
        _id=1,
        name="testname",
        tag="",
        architecture="",
):
    attributes = {
        "id": _id,
        "name": name,
        "tag": tag,
        "digest": "sha256:something",
        "layers_digests": [
            "sha256:layerone",
            "sha256:layertwo"
        ],
        "architecture": architecture
    }
    relationships = {}
    return {
        "attributes": attributes,
        "relationships": relationships
    }

def singleService(
        _id=1,
        label="companyfoo",
        state=0,
        _available=True,
        cpu_demand=1,
        memory_demand=1024,
        image_digest="image0"
):
    attributes = {
        "id": _id,
        "label": label,
        "state": state,
        "_available": _available,
        "cpu_demand": cpu_demand,
        "memory_demand": memory_demand,
        "image_digest": image_digest,
    }
    relationships = {}
    return {
        "attributes": attributes,
        "relationships": relationships
    }

def singleContainerRegistry(
        _id=1,
        cpu_demand=1,
        memory_demand=1024
):
    attributes = {
        "id": _id,
        "cpu_demand": cpu_demand,
        "memory_demand": memory_demand,
    }
    relationships = {}
    return {
        "attributes": attributes,
        "relationships": relationships
    }

def singleApplication(
        _id=1,
        label="",
):
    attributes = {
        "id":_id,
        "label": label
    }
    relationships = {}

    return {
        "attributes": attributes,
        "relationships": relationships
    }

def singleES(
        _id=1,
        available=True,
        model_name="E5430",
        cpu=8,
        memory=16384,
        disk=131072,
        cpu_demand=0,
        memory_demand=0,
        disk_demand=0,
        coordinates=[0,0],
        max_concurrent_layer_downloads=3,
        active=True,
        power_model_parameters={}
):
    attributes = {
        "id": _id,
        "available": available,
        "model_name": model_name,
        "cpu": cpu,
        "memory": memory,
        "disk": disk,
        "cpu_demand": cpu_demand,
        "memory_demand": memory_demand,
        "disk_demand": disk_demand,
        "coordinates": coordinates,
        "max_concurrent_layer_downloads": max_concurrent_layer_downloads,
        "active": active,
        "power_model_parameters": {
            "max_power_consumption": 265,
            "static_power_percentage": 0.6264
        },
    }
    relationships = {}
    return {
        "attributes": attributes,
        "relationships": relationships
    }


def singleRandomDurationAndIntervalAccessPattern():
    return {}


def singleCircularDurationAndIntervalAccessPattern(
        _id=1,
        user_id=1,
        app_id=1,
):
    attributes = {
        "id":_id,
        "duration_values": [
            "Infinity"
        ],
        "interval_values": [
            0
        ],
        "history": [
            {
                "start": 1,
                "end": 100000,
                "duration": 100000,
                "waiting_time": 0,
                "access_time": 0,
                "interval": 0,
                "next_access": 100000,
            }
        ]
    }
    relationships = {
        "user": {
            "class": "User",
            "id": user_id,
        },
        "app": {
            "class": "Application",
            "id": app_id,
        }
    }
    return {
        "attributes": attributes,
        "relationships": relationships,
    }



def index_to_location(index, base=10):
    """
    index to location, first is x coordinate, second is y coordinate
    0 - (0,0), 1 - (1,0), and so on
    """
    return int(index % base), int(index // base)

def location_to_index(location, base):
    x, y = location
    return int(x + y * base)

def location_valid(location, base):
    return location[0] < base and location[1] < base

def main():
    number_of_services = 10
    number_of_applications = 10
    number_of_container_images = 10
    number_of_container_layers = 10
    number_of_container_registries = 10
    base = 10
    # one one relation

    services = [singleService(_id=i,image_digest=f"image{i}") for i in range(number_of_services)]
    applications = [singleApplication(_id=i) for i in range(number_of_applications)]
    container_images = [singleContainerImage(_id=i) for i in range(number_of_container_images)]
    container_layers = [singleContainerLayer(_id=i) for i in range(number_of_container_layers)]
    # TODO
    container_registries = [singleContainerRegistry(_id=i) for i in range(number_of_container_registries)]

    number_of_images = 10
    number_of_layers = 10
    # one one relation

    images = [singleContainerImage(_id=i) for i in range(number_of_images)]
    layers = [singleContainerLayer(_id=i) for i in range(number_of_layers)]

    number_of_users = 1000
    number_of_base_stations = 100
    number_of_network_switches = 100
    number_of_edge_servers = 10
    # 使用的是欧拉定理
    number_of_network_links = number_of_base_stations ** 2 + (number_of_base_stations - 1) ** 2 - 1

    users = [singleUser(_id=i) for i in range(number_of_users)]
    base_stations = [singleBS(_id=i) for i in range(number_of_base_stations)]
    network_switches = [singleNS(_id=i) for i in range(number_of_network_switches)]
    edge_servers  = [singleES(_id=i) for i in range(number_of_edge_servers)]
    network_links = []

    # coordinates or locations
    # bs
    # ns
    # es
    # users
    for idx, bs in enumerate(base_stations):
        bs["attributes"]["coordinates"] = index_to_location(idx)
    for idx, ns in enumerate(network_switches):
        ns["attributes"]["coordinates"] = index_to_location(idx)
    indice_for_es = [77, 64, 75, 90, 7, 56, 52, 70, 33, 89]
    for idx, location_index in enumerate(indice_for_es):
        edge_servers[idx]["attributes"]["coordinates"] = index_to_location(location_index)
    for idx, user in enumerate(users):
        user["attributes"]["coordinates"] = index_to_location(idx // 10)


    ## starting for relations
    ## relation for ns
    for i, ns in enumerate(network_switches):
        ns["relationships"]["power_model"] = "ConteratoNetworkPowerModel"
        ns["relationships"]["edge_servers"] = [] # TO be filled
        ns["relationships"]["links"] = []
        ns["relationships"]["base_station"] = {
            "class": "BaseStation",
            "id": i
        }
    ## TODO relation for NL
    ## relation for bs
    for i, bs in enumerate(base_stations):
        bs["relationships"]["users"] = []        # TO be filled
        bs["relationships"]["edge_servers"] = [] # TO be filled
        bs["relationships"]["network_switch"] = {
            "class": "NetworkSwitch",
            "id": i
        }

    nl_idx = 0
    network_links = []
    for i, ns in enumerate(network_switches):
        location = ns["attributes"]["coordinates"]
        location_right = [location[0]+1, location[1]]
        location_up = [location[0], location[1] + 1]
        if location_valid(location_up, base):
            nl = singleNL(nl_idx)
            nl_idx += 1
            nl["relationships"]["nodes"][0]["id"] = i
            nl["relationships"]["nodes"][1]["id"] = location_to_index(location_up, base=base)
            network_links.append(nl)
        if location_valid(location_right, base):
            nl = singleNL(nl_idx)
            nl_idx += 1
            nl["relationships"]["nodes"][0]["id"] = i
            nl["relationships"]["nodes"][1]["id"] = location_to_index(location_right, base=base)
            network_links.append(nl)

    # TODO relation for user
    # TODO relation for container layer
    # TODO relation for container image
    # TODO relation for service
    # TODO relation for registry
    # DONE relation for application
    # TODO relation for edge servers
    # TODO "RandomDurationAndIntervalAccessPattern"
    # TODO "CircularDurationAndIntervalAccessPattern"

    # relation for edge servers
    indice_for_es = [77, 64, 75, 90, 7, 56, 52, 70, 33, 89]
    for idx, bs_ns_id in enumerate(indice_for_es):
        edge_servers[idx]["relationships"]["power_model"] = "LinearServerPowerModel"
        edge_servers[idx]["relationships"]["base_station"] = {
            "class": "BaseStation",
            "id": bs_ns_id,
        }
        edge_servers[idx]["relationships"]["network_switch"] = {
            "class": "NetworkSwitch",
            "id": bs_ns_id,
        }
        edge_servers[idx]["relationships"]["services"] = [] # TO be filled
        edge_servers[idx]["relationships"]["container_layers"] = [] # TO be filled
        edge_servers[idx]["relationships"]["container_images"] = [] # TO be filled
        edge_servers[idx]["relationships"]["container_registries"] = [] # TO be filled

        # backwards
        base_stations[bs_ns_id]["relationships"]["edge_servers"].append({
            "class": "EdgeServer",
            "id": idx,
        })
        network_switches[bs_ns_id]["relationships"]["edge_servers"].append({
            "class": "EdgeServer",
            "id": idx,
        })


    # relation for application
    for i, app in enumerate(applications):
        app["relationships"]["services"] = []
        app["relationships"]["users"] = [] # TO be filled


        app["relationships"]["services"].append({
            "class": "Service",
            "id": i,
        })



    for i, service in enumerate(services):
        service["relationships"]["application"] = {
            "class": "Application",
            "id": i
        }
        service["relationships"]["server"] = {
            "class": "EdgeServer",
            "id": i
        }
        # backwards
        edge_servers[i]["relationships"]["services"].append({
            "class": "Service",
            "id": i
        })

    for i, layer in enumerate(container_layers):
        # digest
        layer["attributes"]["digest"] = "layer" + str(i)
        # relation
        layer["relationships"]["server"] = {
            "class": "EdgeServer",
            "id": i
        }

    for i, image in enumerate(container_images):
        # attribute
        image["attributes"]["digest"] = "image" + str(i)
        image["attributes"]["layers_digests"] = ["layer" + str(i)]
        # relation
        image["relationships"]["server"] = {
            "class": "EdgeServer",
            "id": i,
        }
    for i, registry in enumerate(container_registries):
        registry["relationships"]["server"] = {
            "class": "EdgeServer",
            "id": i,
        }
    for i, user in enumerate(users):
        user["relationships"]["mobility_model"] = "random"
        random_application_id = random.randint(0,9)
        user["attributes"]["making_requests"][f"{random_application_id}"] = user["attributes"]["making_requests"][f"{i}"]
        user["relationships"]["applications"] = [{
            "class": "Application",
            "id": random_application_id,
        }]
        user["relationships"]["access_patterns"] = {
            f"{random_application_id}": {
                "class": "CircularDurationAndIntervalAccessPattern",
                "id": i,        # user id
            }
        }
        applications[random_application_id]["relationships"]["users"].append({
            "class": "User",
            "id": i
        })


        base_station_idx = int(i // base)
        user["relationships"]["base_station"] = {
            "class": "BaseStation",
            "id": base_station_idx
        }
        base_stations[base_station_idx]["relationships"]["users"].append({
            "class": "User",
            "id": i,
        })

    # for the containerlayer and container image inside of
    # edge servers
    for idx, es in enumerate(edge_servers):
        es["relationships"]["container_layers"].append({
            "class": "ContainerLayer",
            "id": idx           # this means that number of the servers and the layers are the same
        })
        es["relationships"]["container_images"].append({
            "class": "ContainerImage",
            "id": idx,
        })
        es["relationships"]["container_registries"].append({
            "class": "ContainerRegistry",
            "id": idx,
        })


    # for every user, application pair
    # there is a access_pattern
    circular_duration_and_interval_access_patterns = \
        [singleCircularDurationAndIntervalAccessPattern(
            _id=i,
            user_id=i,
            app_id=users[i]["relationships"]["applications"][0]["id"]
        )
         for i in range(number_of_users)]

    return {
        "NetworkSwitch": network_switches,
        "NetworkLink": network_links,
        "BaseStation": base_stations,
        "User": users,
        "ContainerLayer": container_layers,
        "ContainerImage": container_images,
        "Service": services,
        "ContainerRegistry": container_registries,
        "Application": applications,
        "EdgeServer": edge_servers,
        "RandomDurationAndIntervalAccessPattern": [], # as the sample json
        "CircularDurationAndIntervalAccessPattern": circular_duration_and_interval_access_patterns,
    }


data = main()
file_path = "test_config.json"
# file_path = "./test/edgesimpy-tutorials/datasets/sample_dataset1.json"
with open(file_path, "w")  as json_file:
    json.dump(data, json_file, indent=4)
print(f"Write json file to test_config.json")
