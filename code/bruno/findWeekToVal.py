import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import xgboost as xgb
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

path = "../../input/"
path2 = "../../output/"

train_df = pd.read_csv(path+"Trade.csv")

print train_df[(train_df['TradeDateKey'] < 20180421) & (train_df['TradeDateKey'] > 20180415)]
