from pymoo.indicators.hv import HV
import numpy as np
import matplotlib.pyplot as plt

fronts = [np.load(f"mip_front_{i}.npy") for i in range(1,4)]
fronts_mip = [np.load(f"mip_front_{i}.npy") for i in range(1,4)]
list_of_xs_ys = []
for front in fronts:
    front_new = [(point[0], point[1]) for point in front]
    front_new = sorted(front_new)
    # xs, ys = zip(*front_new)
    list_of_xs_ys.append(list(zip(*front_new)))

# list_of_xs_ys = [list(map(list, zip(*fronts[i]))) for i in range(3)]

x_max = max(max(list_of_xs_ys[i][0]) for i in range(3))
y_max = max(max(list_of_xs_ys[i][1]) for i in range(3))

ref_point = np.array([x_max * 1.1, y_max * 1.1])
ind = HV(ref_point=ref_point)

plt.scatter([ref_point[0]], ref_point[1])
for i in range(3):
    xs, ys = list_of_xs_ys[i]
    plt.plot(xs, ys, label=f"cMIP-L{i+1} HV: {ind(fronts[i])}")

front = np.load('nsga_pymoo_fronts.npy')
front_new = [(point[0], point[1]) for point in front]
front_new = sorted(front_new)
xs, ys = zip(*front_new)
plt.plot(xs, ys, label=f"Proposed Method HV: {ind(front)}")

# front = np.load('moea_pymoo.npy')
# front_new = [(point[0], point[1]) for point in front]
# front_new = sorted(front_new)
# xs, ys = zip(*front_new)
# plt.plot(xs, ys, label=f"Proposed Method HV: {ind(front)}")


plt.xlabel("balance")
plt.ylabel("delay")
plt.title("Number of servers: 20")

plt.legend()
plt.grid(True)
plt.show()
