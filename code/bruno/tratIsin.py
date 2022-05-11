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
#dateMin = pd.read_csv(path+"dateMin.csv")
submission = pd.read_csv(path+"sample_submission.csv")
valid_df = pd.read_csv(path+"valid.csv")

isin_df = pd.read_csv(path+"Isin.csv")
mm_df['USD'] = 1
mm_df['ARO'] = mm_df['ARS']
mm_df['INR'] = mm_df['EUR']
mm_df['FRF'] = mm_df['EUR']
mm_df['ITL'] = mm_df['EUR']
mm_df['DEM'] = mm_df['EUR']
mm_df['CNH'] = mm_df['CNY']
for i, r in isin_df.iterrows():
    isin_df['IssuedAmount'][isin_df['IsinIdx'] == i] = isin_df['IssuedAmount'][isin_df['IsinIdx'] == i] * mm_df[r['Currency']].mean()
isin_df.to_csv(path2+"isinTrat.csv")