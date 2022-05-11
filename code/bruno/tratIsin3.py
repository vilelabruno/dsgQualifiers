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

del isin_df['ActualMaturityDateKey'],  isin_df['IssueDateKey']

isin_df['yNr'] = 0
isin_df['yNr'][isin_df['CompositeRating'] == 'NR'] = 1
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'NR'] = 0
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'D'] = 1
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'DD+'] = 2
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'DDD'] = 3
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'DDD+'] = 4
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'C'] = 5
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'C+'] = 6
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'CC-'] = 7
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'CC'] = 8
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'CC+'] = 9
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'CCC-'] = 10
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'CCC'] = 11
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'CCC+'] = 12
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'B-'] = 13
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'B'] = 14
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'B+'] = 15
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'BB-'] = 16
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'BB'] = 17
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'BB+'] = 18
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'BBB-'] = 19
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'BBB'] = 20
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'BBB+'] = 21
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'A-'] = 22
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'A'] = 23
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'A+'] = 24
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'AA-'] = 25
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'AA'] = 26
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'AA+'] = 27
isin_df['CompositeRating'][isin_df['CompositeRating'] == 'AAA'] = 28
isin_df['CompositeRating'] = isin_df['CompositeRating'].astype(int)

#mm_df['USD'] = 1
#mm_df['ARO'] = mm_df['ARS']
#mm_df['INR'] = mm_df['EUR']
#mm_df['FRF'] = mm_df['EUR']
#mm_df['ITL'] = mm_df['EUR']
#mm_df['DEM'] = mm_df['EUR']
#mm_df['CNH'] = mm_df['CNY']
#for i, r in isin_df.iterrows():
#    isin_df['IssuedAmount'][isin_df['IsinIdx'] == i] = isin_df['IssuedAmount'][isin_df['IsinIdx'] == i] * mm_df[r['Currency']].mean()

isin_df.to_csv(path+"isinTratv3.csv")