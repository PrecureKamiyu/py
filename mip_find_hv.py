from pymoo.indicators.hv import HV
import numpy as np

ref_point = np.array([1.1, 1.1])
ind = HV(ref_point=ref_point)

for i in range(1,4):
    front = np.load(f"mip_front_{i}.npy")
    print("number", i, "hv is", ind(front))
