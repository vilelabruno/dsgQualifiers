import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
def extract_precedent_statistics(df, on, group_by):
    
    df.sort_values(group_by + ['visit_date'], inplace=True)
    
    groups = df.groupby(group_by, sort=False)
    
    stats = {
        'mean': [],
        'median': [],
        'std': [],
        'count': [],
        'max': [],
        'min': []
    }
    
    exp_alphas = [0.1, 0.25, 0.3, 0.5, 0.75]
    stats.update({'exp_{}_mean'.format(alpha): [] for alpha in exp_alphas})
    
    for _, group in groups:
        
        shift = group[on].shift()
        roll = shift.rolling(window=len(group), min_periods=1)
        
        stats['mean'].extend(roll.mean())
        stats['median'].extend(roll.median())
        stats['std'].extend(roll.std())
        stats['count'].extend(roll.count())
        stats['max'].extend(roll.max())
        stats['min'].extend(roll.min())
        
        for alpha in exp_alphas:
            exp = shift.ewm(alpha=alpha, adjust=False)
            stats['exp_{}_mean'.format(alpha)].extend(exp.mean())
    
    suffix = '_&_'.join(group_by)
    
    for stat_name, values in stats.items():
        df['{}_{}_by_{}'.format(on, stat_name, suffix)] = values

#%%

path = "../../input/"
path2 = "../../output/"

test_df = pd.read_csv(path+"Challenge_20180423.csv")
train_df = pd.read_csv(path+"train_completo.csv")
#dateMin = pd.read_csv(path+"dateMin.csv")
submission = pd.read_csv(path+"sample_submission.csv")
valid_df = pd.read_csv(path+"valid.csv")
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
isin_df = pd.read_csv(path+"Isin.csv")
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
categorical_feats = [f for f in isin_df.columns if isin_df[f].dtype == 'object']
all_cat = categorical_feats
for i in categorical_feats:
    if (len(isin_df[i].unique()) > 00):
        isin_df[i], indexer = pd.factorize(isin_df[i])
    else:
        ohe = pd.get_dummies(isin_df[i])
        columns = ohe.columns
        for j in columns:
            isin_df[j] = ohe[j]
        isin_df.drop(i, axis=1, inplace=True)
categorical_feats = [f for f in isin_df.columns if isin_df[f].dtype == 'object']
print(categorical_feats)

# --> Categorical Customer <--
cost_df = pd.read_csv(path+"Customer.csv")
del cost_df['Region']
categorical_feats = [f for f in cost_df.columns if cost_df[f].dtype == 'object']
for i in categorical_feats:
    if (len(cost_df[i].unique()) > 0):
        cost_df[i], indexer = pd.factorize(cost_df[i])
    else:
        ohe = pd.get_dummies(cost_df[i])
        columns = ohe.columns
        for j in columns:
            cost_df[j] = ohe[j]
        cost_df.drop(i, axis=1, inplace=True)
categorical_feats = [f for f in cost_df.columns if cost_df[f].dtype == 'object']
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

# --> Model <--
d_train = xgb.DMatrix(train_df, y_train)
d_valid = xgb.DMatrix(valid_df, y_valid)

#%%
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#watchlist = [(d_train, 'train')]

params = {
    'eta': 0.1,
    'max_depth': 10, 
    'subsample': 0.9, 
    'colsample_bytree': 0.9, 
    'colsample_bylevel':0.9,
    #'min_child_weight':10,
    #'alpha':4,
    'objective': 'binary:logistic',
    'eval_metric': 'auc', 
    'nthread':8,
    'random_state': 99, 
    'silent': True}
    
model = xgb.train(params, d_train, 300, watchlist, 
                  maximize=True, early_stopping_rounds = 20, 
                  verbose_eval=1)


#%%

preds = model.predict(xgb.DMatrix(test_df))
submission['CustomerInterest'] = preds

submission.to_csv(path2+"v_005.csv", index=False)

#####