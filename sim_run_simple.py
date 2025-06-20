from edge_sim_py import *
import networkx as nx
import msgpack

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

input_file = "/home/one/test/edgesimpy-tutorials/datasets/sample_dataset1.json"
# input_file = "test_config.json"
from sim_test_service_placement import my_algorithm, stopping_criterion
simulator = Simulator(
    dump_interval=5,
    tick_duration=1,
    tick_unit='seconds',
    stopping_criterion=stopping_criterion,
    resource_management_algorithm=my_algorithm,
)
simulator.initialize(input_file=input_file)
print("Done Initilazation")


print("Start simulation")
simulator.run_model()
