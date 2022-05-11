import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

path = "../../input/"
path2 = "../../output/"

test_df = pd.read_csv(path+"Challenge_20180423.csv")
train_df = pd.read_csv(path+"Trade.csv")
#train_df.drop('Day', axis=1, inplace=True)

#%%

train_df['DateTime'] = pd.to_datetime(train_df['TradeDateKey'], format='%Y%m%d')
train_df = train_df[(train_df['TradeDateKey'] < 20180101) & (train_df['TradeDateKey'] >= 20170101)]

last_days = [20160131,20160229,20160331,20160430,20160531,20160630,20160731,20160831,20160930,20161031,20161130,20161231,
             20170131,20170228,20170331,20170430,20170531,20170630,20170731,20170831,20170930,20171031,20171130,20171231,
             20180131,20180228,20180331]

train_df = train_df[~train_df['TradeDateKey'].isin(last_days)] 

train_df['WeekOfYear'] = train_df['DateTime'].dt.week
train_df['Year'] = train_df['DateTime'].dt.year

aux2 = train_df[train_df['TradeDateKey'] <= 20180415]
aux = aux2.groupby(['CustomerIdx','IsinIdx'])['TradeDateKey'].count().reset_index()
aux = aux.rename(columns={'TradeDateKey': 'count'})
#%%
initial_days = train_df.groupby(['Year','WeekOfYear'])['TradeDateKey'].min().reset_index()
initial_days.columns = ['Year','WeekOfYear', 'DateKey']

final_days = train_df.groupby(['Year','WeekOfYear'])['TradeDateKey'].max().reset_index()
final_days.columns = ['Year','WeekOfYear', 'TradeDateKey']

cust_initial_dates = train_df.groupby(['CustomerIdx','IsinIdx'])['TradeDateKey'].min().reset_index()

del train_df['DateTime'], train_df['TradeDateKey'], train_df['NotionalEUR'], train_df['TradeStatus'], train_df['Price']

k = train_df.drop_duplicates()

all_combinations = test_df.groupby(['CustomerIdx','IsinIdx'])['CustomerInterest'].count().reset_index()
del all_combinations['CustomerInterest']

all_combinations = pd.merge(all_combinations, cust_initial_dates, on=['CustomerIdx','IsinIdx'], how='left', sort=False)
all_combinations['TradeDateKey'] = all_combinations['TradeDateKey'].fillna(20160101)

#%%

all_buys = all_combinations.copy()
all_buys['BuySell'] = 'Buy'

all_sells = all_combinations.copy()
all_sells['BuySell'] = 'Sell'

all_combinations = pd.concat([all_buys, all_sells])

#%%

all_combinations_2 = pd.DataFrame()

all_weeks_18 = k['WeekOfYear'][k['Year'] == 2018].unique()
all_weeks_17 = k['WeekOfYear'][k['Year'] == 2017].unique()

final_days_18 = final_days[final_days['Year'] == 2018]
final_days_17 = final_days[final_days['Year'] == 2017]

for week in all_weeks_18:
    last_day = final_days_18['TradeDateKey'][final_days_18['WeekOfYear'] == week]
    aux_all = all_combinations.copy()
    aux_all = aux_all[aux_all['TradeDateKey'] <= np.int(last_day)]
    aux_all['Year'] = 2018
    aux_all['WeekOfYear'] = week
    all_combinations_2 = pd.concat([all_combinations_2, aux_all])

for week in all_weeks_17:
    last_day = final_days_17['TradeDateKey'][final_days_17['WeekOfYear'] == week]
    aux_all = all_combinations.copy()
    aux_all = aux_all[aux_all['TradeDateKey'] <= np.int(last_day)]
    aux_all['Year'] = 2017
    aux_all['WeekOfYear'] = week
    all_combinations_2 = pd.concat([all_combinations_2, aux_all])
        

#%%

all_combinations_2 = pd.merge(all_combinations_2, k, on=['CustomerIdx','IsinIdx','BuySell','Year','WeekOfYear'], how='left', sort=False)
all_combinations_2['CustomerInterest'] = all_combinations_2['CustomerInterest'].fillna(0)

k = all_combinations_2.copy()

k = pd.merge(k, initial_days, on=['Year','WeekOfYear'], how='left', sort=False)
k = pd.merge(k, aux, on=['CustomerIdx','IsinIdx'], how='left', sort=False)

del k['Year'], k['WeekOfYear'], k['TradeDateKey']

k.to_csv(path+"train_count.csv", index=False)

#%%
