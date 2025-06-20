from edge_sim_py import *

def my_algorithm(parameters):
    for service in Service.all():
        if service.server == None and not service.being_provisioned:
            for edge_server in EdgeServer.all():
                if edge_server.has_capacity_to_host(service=service):
                    service.provision(target_server=edge_server)
                    break


def stopping_criterion(model: object):
    provisioned_services = 0
    for service in Service.all():
        if service.server != None:
            provisioned_services += 1
    return provisioned_services == Service.count()
