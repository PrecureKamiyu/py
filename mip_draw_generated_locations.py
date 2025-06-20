# * For the randomly generated locations, we draw scatter graph
import matplotlib.pyplot as plt
import numpy as np


def draw_locations_to_be_chosen(index=1, path=''):
    i = index

    locations = np.load(f"locations_to_be_chosen_{i}.npy")
    xs = [location[0] for location in locations]
    ys = [location[1] for location in locations]
    plt.scatter(xs,ys)
    plt.grid(True)
    plt.show()

    if path != '':
        plt.savefig(path)



if __name__ == "__main__":
    draw_locations_to_be_chosen(index=1)
