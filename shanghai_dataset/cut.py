import pandas as pd

df = pd.read_csv('./data_0601_to_0615.csv')
new_df = df[['latitude', 'longitude']]
new_df.to_csv('data_0601_to_0615_location.csv', index=False)
