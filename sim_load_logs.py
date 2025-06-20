import os
import msgpack
import pandas as pd

# Gathering the list of msgpack files in the current directory
logs_directory = f"{os.getcwd()}/logs"
dataset_files = [file for file in os.listdir(logs_directory) if ".msgpack" in file]

# Reading msgpack files found
datasets = {}
for file in dataset_files:
    with open(f"logs/{file}", "rb") as data_file:
        datasets[file.replace(".msgpack", "")] = pd.DataFrame(msgpack.unpackb(data_file.read(), strict_map_key=False))

print(datasets["EdgeServer"])
print(datasets["User"])
print(datasets["NetworkSwitch"])
