import alphashape
import numpy as np
import pandas as pd
from descartes import PolygonPatch
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('./shanghai_dataset/block_counts.csv')
    lat = np.array(df['lat_block'])
    lon = np.array(df['lon_block'])

    points = list(zip(lon, lat))
    alpha = 4.0
    alpha_shape = alphashape.alphashape(points, alpha)
    fig, ax = plt.subplots()
    ax.scatter(*zip(*points))
    ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))
    plt.show()

main()
