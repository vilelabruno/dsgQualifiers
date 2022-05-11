import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
path = "../../input/"
path2 = "../../output/"

test_df = pd.read_csv(path+"Challenge_20180423.csv")
train_df = pd.read_csv(path+"trainDropVal.csv")
mm_df = pd.read_csv(path+"mmOnlyConversion.csv")
isin_df = pd.read_csv(path+"Isin.csv")
#dateMin = pd.read_csv(path+"dateMin.csv")
submission = pd.read_csv(path+"sample_submission.csv")
valid_df = pd.read_csv(path+"valid.csv")

aux1 = pd.to_datetime(isin_df['ActualMaturityDateKey'], format='%Y%m%d') 
aux2 = pd.to_datetime('20180423', format='%Y%m%d') 
aux3 = pd.to_datetime(isin_df['IssueDateKey'], format='%Y%m%d') 

isin_df['diffMatToEm'] = aux1 - aux3
isin_df['diffMatToNow'] = aux1 - aux2
isin_df['diffEmToNow'] = aux2 - aux3
isin_df['diffMatToNow'] = isin_df['diffMatToNow'].dt.days
isin_df['diffMatToNow'] = isin_df['diffMatToNow'].astype(int)
isin_df['diffEmToNow'] = isin_df['diffEmToNow'].dt.days
isin_df['diffEmToNow'] = isin_df['diffEmToNow'].astype(int)
isin_df['diffMatToEm'] = isin_df['diffMatToEm'].dt.days
isin_df['diffMatToEm'] = isin_df['diffMatToEm'].astype(int)



isin_df.to_csv(path+"isinTrat2.csv")