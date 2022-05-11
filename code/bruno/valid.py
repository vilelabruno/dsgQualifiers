import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

path = "/home/bruno/dsg_2018/input/"
path2 = "/home/bruno/dsg_2018/output/"

test_df = pd.read_csv(path+"Challenge_20180423.csv")
train_df = pd.read_csv(path+"Trade.csv")
#train_df.drop('Day', axis=1, inplace=True)
aux2 = train_df[train_df['TradeDateKey'] <= 20180415]
aux = aux2.groupby(['CustomerIdx','IsinIdx'])['TradeDateKey'].count().reset_index()
aux = aux.rename(columns={'TradeDateKey': 'count'})
valid_aux = train_df[(train_df['TradeDateKey'] >= 20171016) & (train_df['TradeDateKey'] < 20180416)]

last_days = [20160131,20160229,20160331,20160430,20160531,20160630,20160731,20160831,20160930,20161031,20161130,20161231,
             20170131,20170228,20170331,20170430,20170531,20170630,20170731,20170831,20170930,20171031,20171130,20171231,
             20180131,20180228,20180331]

valid_aux = valid_aux[~valid_aux['TradeDateKey'].isin(last_days)] 

last_week = train_df[train_df['TradeDateKey'] >= 20180416]

del last_week['TradeDateKey'], last_week['NotionalEUR'], last_week['TradeStatus'], last_week['Price']

last_week = last_week.drop_duplicates()
last_week = last_week.groupby(['CustomerIdx','IsinIdx','BuySell'])['CustomerInterest'].sum().reset_index()
#last_week['CustomerInterest'] = 1

last_6M_comb = valid_aux.groupby(['CustomerIdx','IsinIdx'])['CustomerInterest'].count().reset_index()
del last_6M_comb['CustomerInterest']

last_6M_comb_buy = last_6M_comb.copy()
last_6M_comb_buy['BuySell'] = 'Buy'
last_6M_comb_sell = last_6M_comb.copy()
last_6M_comb_sell['BuySell'] = 'Sell'

valid = pd.concat([last_6M_comb_buy, last_6M_comb_sell])

valid = pd.merge(valid, last_week, on=['CustomerIdx','IsinIdx','BuySell'], how='left', sort=False)
valid['CustomerInterest'] = valid['CustomerInterest'].fillna(0)

valid['DateKey'] = 20180416

valid = pd.merge(valid, aux, on=['CustomerIdx','IsinIdx'], how='left')

valid.to_csv(path+"valid_count.csv",index=False)
