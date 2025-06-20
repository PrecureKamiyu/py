import pandas as pd
import numpy as np
import math

def distance(p1, p2):
    """
    Return Distance of two given points.
    Args:
      they are all two tuples
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def foo1(location, diameter, block_counts):
    """
    Args:
      location (two tuple): fst is lattitute and snd is longitute
    Return:
      float: the objective value.
    """
    lat = block_counts['lat_block']
    lon = block_counts['lon_block']
    counts = block_counts['count']

    ret = 0
    for i in range(len(lat)):
        location_i = [lat[i], lon[i]]
        d = math.dist(location_i, location)
        if d < diameter:
            ret += counts[i] * d
    return (ret / counts) / diameter


def foo2(locations, diameter, block_counts):
    """
    Args:
      location (two tuple): fst is lattitute and snd is longitute
    Return:
      float: the objective value.
    """
    lat = block_counts['lat_block']
    lon = block_counts['lon_block']
    counts = block_counts['count']

    # s1 is now \bar W
    s1 = 0.0
    for location in locations:
        s2 = 0.0
        for i in range(len(lat)):
            location_i = [lat[i], lon[i]]
            d = math.dist(location_i, location)
            if d < diameter:
                s2 += counts[i]
        s1 += s2
    s1 = s1 / len(locations)
