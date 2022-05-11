import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

path = "../../input/"
path2 = "../../output/"

test_df = pd.read_csv(path+"Challenge_20180423.csv")
train_df = pd.read_csv(path+"Trade.csv")

last_week = train_df[train_df['TradeDateKey'] >= 20180401]

last_week = last_week[['CustomerIdx','IsinIdx','BuySell','CustomerInterest']]
last_week = last_week.drop_duplicates()
last_week = last_week.groupby(['CustomerIdx','IsinIdx','BuySell'])['CustomerInterest'].sum().reset_index()

valid = test_df.copy()
valid.drop("CustomerInterest", axis=1, inplace=True)
valid = pd.merge(valid, last_week, on=['CustomerIdx','IsinIdx','BuySell'], how='left', sort=False)

valid['CustomerInterest'] = valid['CustomerInterest'].fillna(0)

last_week['Aux'] = 1
del last_week['CustomerInterest'], last_week['BuySell']

valid = pd.merge(valid, last_week, on=['CustomerIdx','IsinIdx'], how='left', sort=False)
valid = valid[valid['Aux'] == 1]
del valid['Aux']

valid['DateKey'] = 20180401

valid.to_csv(path+"valid.csv", index=False)

#%%