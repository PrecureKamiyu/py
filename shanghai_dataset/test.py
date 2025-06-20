# this is the realest one
import pandas as pd
import numpy as py

csv_path = 'data_0601_to_0615.csv'
# csv_path = "test_0601.csv"
df = pd.read_csv(csv_path)
df = df.dropna()

df = df[
    (df['month'] == 201406) & \
    (df['date']  == 1)
]

df['start time'] = df['start time'].map(lambda s: s[-8:])
df['arrival time'] = pd.to_timedelta(df['start time']).dt.total_seconds().astype(int)

df['lat'] = df['latitude'] - 31.0
df['lon'] = df['longitude'] - 121.0
df = df[
    (df['lat'] < 1) & (df['lat'] >= 0) & \
    (df['lon'] < 1) & (df['lon'] >= 0)
]
df['lat'] = df['lat'] * 10
df['lon'] = df['lon'] * 10

df['lat'] = df['lat'].map(int)
df['lon'] = df['lon'].map(int)



codes, uniques = pd.factorize(df['user id'])
df['user_id'] = codes

df['bs_id_adhoc'] = df['lat'] + df['lon'] * 10

print(df['arrival time'])
df.to_csv('test_0601.csv', index=False)
df.to_csv('0601.csv', index=False)
print(df)
