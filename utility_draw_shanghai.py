import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('./shanghai_dataset/block_counts.csv')
    lat = np.array(df['lat_block'])
    lon = np.array(df['lon_block'])
    plt.scatter(lon, lat)
    plt.grid(True)
    plt.show()


main()
