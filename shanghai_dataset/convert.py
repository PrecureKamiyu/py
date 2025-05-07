import pandas as pd
df = pd.read_excel('data_6.1~6.15.xlsx')
df.to_csv('data_0601_to_0615.csv', index=False)
