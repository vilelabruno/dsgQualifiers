import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

path = "../../input/"
path2 = "../../output/"

test_df = pd.read_csv(path+"Challenge_20180423.csv")
train_df = pd.read_csv(path+"Trade.csv")

#%%

train_df['DateTime'] = pd.to_datetime(train_df['TradeDateKey'], format='%Y%m%d')
#train_df = train_df[train_df['TradeDateKey'] < 20180401]

last_days = [20160131,20160229,20160331,20160430,20160531,20160630,20160731,20160831,20160930,20161031,20161130,20161231,
             20170131,20170228,20170331,20170430,20170531,20170630,20170731,20170831,20170930,20171031,20171130,20171231,
             20180131,20180228,20180331]

train_df = train_df[~train_df['TradeDateKey'].isin(last_days)] 

train_df['WeekOfYear'] = train_df['DateTime'].dt.week
train_df['Year'] = train_df['DateTime'].dt.year

#%%
del train_df['DateTime'], train_df['TradeDateKey'], train_df['NotionalEUR'], train_df['TradeStatus'], train_df['Price']

k = train_df.drop_duplicates()

all_combinations = k.groupby(['CustomerIdx','IsinIdx','Year','WeekOfYear'])['CustomerInterest'].sum().reset_index()

del all_combinations['CustomerInterest']

all_buys = all_combinations.copy()
all_buys['BuySell'] = 'Buy'

all_sells = all_combinations.copy()
all_sells['BuySell'] = 'Sell'

all_combinations = pd.concat([all_buys, all_sells])

all_combinations = pd.merge(all_combinations, k, on=['CustomerIdx','IsinIdx','BuySell','Year','WeekOfYear'], how='left', sort=False)
all_combinations['CustomerInterest'] = all_combinations['CustomerInterest'].fillna(0)

k = all_combinations.copy()

del k['Year'], k['WeekOfYear'], 

k.to_csv(path+"new_train.csv", index=False)

#%%
