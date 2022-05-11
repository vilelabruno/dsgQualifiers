import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import xgboost as xgb
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

#%%

path = "../../input/"
path2 = "../../output/"

test_df = pd.read_csv(path+"Challenge_20180423.csv")
train_df = pd.read_csv(path+"new_train.csv")
submission = pd.read_csv(path+"sample_submission.csv")
valid_df = pd.read_csv(path+"valid.csv")

#%%

# --> Train Treatment <--
#columns = ["NotionalEUR", "Price", "TradeStatus"]
#for k in columns:
#    train_df.drop(k, axis=1, inplace=True)
    
ohe = pd.get_dummies(train_df["BuySell"])
columns = ohe.columns
for j in columns:
    train_df[j] = ohe[j]
train_df.drop("BuySell", axis=1, inplace=True)

valid_df.drop("PredictionIdx", axis=1, inplace=True)
ohe = pd.get_dummies(valid_df["BuySell"])
columns = ohe.columns
for j in columns:
    valid_df[j] = ohe[j]
valid_df.drop("BuySell", axis=1, inplace=True)

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
categorical_feats = [f for f in isin_df.columns if isin_df[f].dtype == 'object']
for i in categorical_feats:
    if (len(isin_df[i].unique()) > 0):
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
print(categorical_feats)


#%%

# --> Merging <--
train_df = pd.merge(train_df, isin_df, on='IsinIdx', how='left', sort=False)
train_df = pd.merge(train_df, cost_df, on='CustomerIdx', how='left', sort=False)
valid_df = pd.merge(valid_df, isin_df, on='IsinIdx', how='left', sort=False)
valid_df = pd.merge(valid_df, cost_df, on='CustomerIdx', how='left', sort=False)
test_df = pd.merge(test_df, isin_df, on='IsinIdx', how='left', sort=False)
test_df = pd.merge(test_df, cost_df, on='CustomerIdx', how='left', sort=False)


# --> Preparing for Train <--
# =============================================================================
# idx_columns = ['CustomerIdx','IsinIdx','TickerIdx','Country']
# for column in idx_columns:
#     train_df.drop(column, axis=1, inplace=True)
#     valid_df.drop(column, axis=1, inplace=True)
#     test_df.drop(column, axis=1, inplace=True)
# 
# =============================================================================


y_train = train_df['CustomerInterest']
y_valid = valid_df['CustomerInterest']
#test_df['Data'] = test_df['DateKey']
del test_df['DateKey']
del test_df['CustomerInterest']
#train_df['Data'] = train_df['DateKey']
#del train_df['DateKey']
del train_df['CustomerInterest']
#valid_df['Data'] = valid_df['DateKey']
del valid_df['DateKey']
del valid_df['CustomerInterest']

# --> Model <--
d_train = xgb.DMatrix(train_df, y_train)
d_valid = xgb.DMatrix(valid_df, y_valid)

#%%

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
    
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#watchlist = [(d_train, 'train')]
model = xgb.train(params, d_train, 10, watchlist, 
                  maximize=True, early_stopping_rounds = 20, 
                  verbose_eval=1)

xgb.plot_importance(model)

#%%

preds = model.predict(xgb.DMatrix(test_df))
submission['CustomerInterest'] = preds

submission.to_csv(path2+"version_022.csv", index=False)

######