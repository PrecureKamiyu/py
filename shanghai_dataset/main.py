import pandas as pd
import numpy as py

csv_path = 'data_0601_to_0615.csv'

# * processing

df = pd.read_csv(csv_path)

# ** drop nan value

df = df.dropna()

# ** select rows of date June the First

df = df[
    (df['month'] == 201406) & \
    (df['date']  == 1)
]

# ** time string manipulation

# 'start time' is what we need
# and start time is of format
# HH:MM:SS at the last eight characters
# so we can just take last eight characters
df['start time'] = df['start time'].map(lambda s: s[-8:])
# and then use pd.to_timedelta in t
df['arrival time'] = pd.to_timedelta(df['start time']).dt.total_seconds().astype(int)

# ** grid-ify the map

df['lat'] = df['latitude'] - 31.0
df['lon'] = df['longitude'] - 121.0
df = df[
    (df['lat'] < 1) & (df['lat'] >= 0) & \
    (df['lon'] < 1) & (df['lon'] >= 0)
]
df['lat'] = df['lat'] * 10
df['lon'] = df['lon'] * 10
# I seperate them because I thought there
# was some mysterious bug, but now I have
# no interests in verify the existence of the bug
# if it ain't broke don't fix it
df['lat'] = df['lat'].map(int)
df['lon'] = df['lon'].map(int)

# ** turning the user id into sequential id

# 'user id' is a mysterious string
# but I turn it into 'user_id' which is
# just sequential incrementing id
codes, uniques = pd.factorize(df['user id'])
df['user_id'] = codes

# ** find the corresponding block (where there is a base station)

df['bs_id_adhoc'] = df['lat'] + df['lon'] * 10

# * just some printing
print(df['arrival time'])
df.to_csv('test_0601.csv', index=False)
df.to_csv('0601.csv', index=False)
print(df)
