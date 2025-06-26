import itertools as its
import os


ns = [10,20,30]
N = 160

for n in ns:
    command = f"python mip_whole.py {n} --N {N}"
    print("run this command", command)
    os.system(command)
