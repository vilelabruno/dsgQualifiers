# Initialize data
import os
import gc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import roc_auc_score as ras
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
#%%

path = "../../input/"
path2 = "../../output/"

test_df = pd.read_csv(path+"Challenge_20180423.csv")
train_df = pd.read_csv(path+"def_train.csv")
#dateMin = pd.read_csv(path+"dateMin.csv")
submission = pd.read_csv(path+"sample_submission.csv")
valid_df = pd.read_csv(path+"def_valid.csv")
#%%
ohe = pd.get_dummies(train_df["BuySell"])
columns = ohe.columns
for j in columns:
    train_df[j] = ohe[j]
train_df.drop("BuySell", axis=1, inplace=True)

# =============================================================================
ohe = pd.get_dummies(valid_df["BuySell"])
columns = ohe.columns
for j in columns: 
    valid_df[j] = ohe[j]
valid_df.drop("BuySell", axis=1, inplace=True)
# =============================================================================

# --> Test Treatment <-
test_df.drop("PredictionIdx", axis=1, inplace=True)
ohe = pd.get_dummies(test_df["BuySell"])
columns = ohe.columns
for j in columns:
    test_df[j] = ohe[j]
test_df.drop("BuySell", axis=1, inplace=True)


#%%

# --> Categorical Isin <--
isin_df = pd.read_csv(path+"isinTrat2.csv")
#isin_df['yNr'] = 0
#isin_df['yNr'][isin_df['CompositeRating'] == 'NR'] = 1
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'NR'] = 0
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'D'] = 1
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'DD+'] = 2
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'DDD'] = 3
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'DDD+'] = 4
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'C'] = 5
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'C+'] = 6
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'CC-'] = 7
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'CC'] = 8
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'CC+'] = 9
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'CCC-'] = 10
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'CCC'] = 11
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'CCC+'] = 12
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'B-'] = 13
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'B'] = 14
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'B+'] = 15
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'BB-'] = 16
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'BB'] = 17
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'BB+'] = 18
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'BBB-'] = 19
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'BBB'] = 20
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'BBB+'] = 21
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'A-'] = 22
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'A'] = 23
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'A+'] = 24
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'AA-'] = 25
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'AA'] = 26
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'AA+'] = 27
#isin_df['CompositeRating'][isin_df['CompositeRating'] == 'AAA'] = 28
#isin_df['CompositeRating'] = isin_df['CompositeRating'].astype(int)

categorical_feats = [f for f in isin_df.columns if isin_df[f].dtype == 'object']


cat_feats_idx = np.unique([np.argmax(isin_df.columns == feat) for feat in categorical_feats])
all_cat = categorical_feats
for i in categorical_feats:
    if (len(isin_df[i].unique()) > 10):
        isin_df[i], indexer = pd.factorize(isin_df[i])
    else:
        ohe = pd.get_dummies(isin_df[i])
        columns = ohe.columns
        for j in columns:
            isin_df[j] = ohe[j]
        isin_df.drop(i, axis=1, inplace=True)
#categorical_feats = [f for f in isin_df.columns if isin_df[f].dtype == 'object']
print(categorical_feats)

# --> Categorical Customer <--
cost_df = pd.read_csv(path+"Customer.csv")
del cost_df['Region']
categorical_feats = [f for f in cost_df.columns if cost_df[f].dtype == 'object']
for i in categorical_feats:
    if (len(cost_df[i].unique()) > 10):
        cost_df[i], indexer = pd.factorize(cost_df[i])
    else:
        ohe = pd.get_dummies(cost_df[i])
        columns = ohe.columns
        for j in columns:
            cost_df[j] = ohe[j]
        cost_df.drop(i, axis=1, inplace=True)
#categorical_feats += [f for f in cost_df.columns if cost_df[f].dtype == 'object']
all_cat = all_cat + categorical_feats
print(categorical_feats)

print(all_cat)
#%%

# --> Merging <--
train_df = pd.merge(train_df, isin_df, on='IsinIdx', how='left', sort=False)
train_df = pd.merge(train_df, cost_df, on='CustomerIdx', how='left', sort=False)
#train_df = pd.merge(train_df, dateMin, on=['CustomerIdx', 'IsinIdx'], how='left', sort=False)
valid_df = pd.merge(valid_df, isin_df, on='IsinIdx', how='left', sort=False)
valid_df = pd.merge(valid_df, cost_df, on='CustomerIdx', how='left', sort=False)
#valid_df = pd.merge(valid_df, dateMin, on=['CustomerIdx', 'IsinIdx'], how='left', sort=False)
test_df = pd.merge(test_df, isin_df, on='IsinIdx', how='left', sort=False)
test_df = pd.merge(test_df, cost_df, on='CustomerIdx', how='left', sort=False)
#test_df = pd.merge(test_df, dateMin, on=['CustomerIdx', 'IsinIdx'], how='left', sort=False)


#train_df['dateMin'] = pd.to_datetime(train_df['dateMin'], format='%Y%m%d')
#train_df['dateAux'] = pd.to_datetime(train_df['DateKey'], format='%Y%m%d')
#train_df['dateMin'] = train_df['dateAux'] - train_df['dateMin']
#train_df['dateMin'] = train_df['dateMin'].dt.days
#train_df['dateMin'] = train_df['dateMin'].astype(int)
#train_df.drop('dateAux', axis=1, inplace=True)
#
#valid_df['dateMin'] = pd.to_datetime(valid_df['dateMin'], format='%Y%m%d')
#valid_df['dateAux'] = pd.to_datetime(valid_df['DateKey'], format='%Y%m%d')
#valid_df['dateMin'] = valid_df['dateAux'] - valid_df['dateMin']
#valid_df['dateMin'] = valid_df['dateMin'].dt.days
#valid_df['dateMin'] = valid_df['dateMin'].astype(int)
#valid_df.drop('dateAux', axis=1, inplace=True)
#
#test_df['dateMin'] = pd.to_datetime(test_df['dateMin'], format='%Y%m%d')
#test_df['dateAux'] = pd.to_datetime(test_df['DateKey'], format='%Y%m%d')
#test_df['dateMin'] = test_df['dateAux'] - test_df['dateMin']
#test_df['dateMin'] = test_df['dateMin'].dt.days
#test_df['dateMin'] = test_df['dateMin'].astype(int)
#test_df.drop('dateAux', axis=1, inplace=True)


train_df = train_df[train_df['DateKey'] != 20180416]


y_train = train_df['CustomerInterest']
y_valid = valid_df['CustomerInterest']
del test_df['DateKey']
del test_df['CustomerInterest']
del train_df['DateKey'] 
del train_df['CustomerInterest']
del valid_df['DateKey']
del valid_df['CustomerInterest']
lgb_train = lgb.Dataset(train_df, y_train, free_raw_data=False)
num_train, num_feature = train_df.shape
lgb_eval = lgb.Dataset(valid_df, y_valid, reference=lgb_train, free_raw_data=False)
# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.15
}
# generate a feature name
feature_name = ['feature_' + str(col) for col in range(num_feature)]

print('Start training...')
# feature_name and categorical_feature
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=500,
                valid_sets=lgb_train,  # eval training data
                feature_name=feature_name)
y_pred = gbm.predict(valid_df)

print (ras(y_valid, y_pred))
