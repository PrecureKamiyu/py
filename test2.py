import matplotlib.pyplot as plt
import numpy as np

nsga = 'nsga_pymoo_fronts.npy'
moea = 'moea_pymoo.npy'

points1 = np.load(nsga)
points2 = np.load(moea)

plt.scatter([point[0] for point in points1], [point[1] for point in points1])
plt.scatter([point[0] for point in points2], [point[1] for point in points2])
plt.grid(True)
plt.show()
