import numpy as np
import matplotlib.pyplot as plt

front = np.load('moea_pymoo.npy')
front_new = [(point[0], 5 * point[1]) for point in front]
front_new = sorted(front_new)
xs, ys = zip(*front_new)

plt.plot(xs, ys)
plt.grid(True)
plt.show()
