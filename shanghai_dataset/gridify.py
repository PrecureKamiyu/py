import pandas as pd

df = pd.read_csv("block_counts.csv")
print(df)

df["lat_block"] = df["lat_block"] * 1000
df["lon_block"] = df["lon_block"] * 1000
df["lat_block"] = df["lat_block"].apply(lambda x: round(x))
df["lon_block"] = df["lon_block"].apply(lambda x: round(x))

