import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score

#%%

path = "../../input/"
path2 = "../../output/"

valid_df = pd.read_csv(path+"Challenge_20180423.csv")
train_df = pd.read_csv(path+"Trade.csv")

aux2 = train_df[train_df['TradeDateKey'] <= 20180415]
aux = aux2.groupby(['CustomerIdx','IsinIdx'])['TradeDateKey'].count().reset_index()
aux = aux.rename(columns={'TradeDateKey': 'count'})

valid_df = pd.merge(valid_df, aux, on=['CustomerIdx','IsinIdx'], how='left')
valid_df.to_csv(path+"test_count.csv",index=False)
