import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import xgboost as xgb
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())


#%%
path = "/home/lpd/Downloads/dsg_2018/input/"
path2 = "/home/lpd/Downloads/dsg_2018/output/"

test_df = pd.read_csv(path+"Challenge_20180423.csv")
train_df = pd.read_csv(path+"Trade.csv")
submission = pd.read_csv(path+"sample_submission.csv")

# --> Train Treatment <--
columns = ["NotionalEUR", "Price", "TradeStatus"]
for k in columns:
    train_df.drop(k, axis=1, inplace=True)
    
ohe = pd.get_dummies(train_df["BuySell"])
columns = ohe.columns
for j in columns:
    train_df[j] = ohe[j]
train_df.drop("BuySell", axis=1, inplace=True)

# --> Test Treatment <-
test_df.drop("PredictionIdx", axis=1, inplace=True)
ohe = pd.get_dummies(test_df["BuySell"])
columns = ohe.columns
for j in columns:
    test_df[j] = ohe[j]
test_df.drop("BuySell", axis=1, inplace=True)

# --> Categorical Isin <--
isin_df = pd.read_csv(path+"Isin.csv")
categorical_feats = [f for f in isin_df.columns if isin_df[f].dtype == 'object']
for i in categorical_feats:
    if (len(isin_df[i].unique()) > 5):
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
    if (len(cost_df[i].unique()) > 5):
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

# --> Bloco de Validaçao <--
LastSellAndBuy = train_df.groupby(['CustomerIdx', 'IsinIdx', 'Buy'])['TradeDateKey'].max().reset_index()
CountBuys = LastSellAndBuy.groupby(['CustomerIdx','IsinIdx'])['Buy'].count().reset_index()
CountBuys.columns = ['CustomerIdx','IsinIdx','CountBuys']
LastSellAndBuy = pd.merge(LastSellAndBuy, CountBuys, on=['CustomerIdx','IsinIdx'], how='left', sort=False)
LastSellAndBuy = LastSellAndBuy[LastSellAndBuy['CountBuys'] > 1]
del LastSellAndBuy['CountBuys']
LastSellAndBuy = LastSellAndBuy[LastSellAndBuy['TradeDateKey'] > 20180101]
valid_df = pd.merge(LastSellAndBuy, train_df, on=['CustomerIdx','IsinIdx', 'Buy', 'TradeDateKey'], how='left', sort=False)
valid_df = valid_df.drop_duplicates()
train_df = train_df[train_df['TradeDateKey'] <= 20180101]

#%%
# --> Merging <--
train_df = pd.merge(train_df, isin_df, on='IsinIdx', how='left', sort=False)
train_df = pd.merge(train_df, cost_df, on='CustomerIdx', how='left', sort=False)
valid_df = pd.merge(valid_df, isin_df, on='IsinIdx', how='left', sort=False)
valid_df = pd.merge(valid_df, cost_df, on='CustomerIdx', how='left', sort=False)
test_df = pd.merge(test_df, isin_df, on='IsinIdx', how='left', sort=False)
test_df = pd.merge(test_df, cost_df, on='CustomerIdx', how='left', sort=False)

# --> Feature Engineering <--
# =============================================================================
# count_Buys = train_df.groupby(['CustomerIdx','IsinIdx'])['Buy'].sum().reset_index()
# count_Buys.columns = ['CustomerIdx','IsinIdx','Buys_Count_UI']
# train_df = pd.merge(train_df, count_Buys, on=['CustomerIdx','IsinIdx'], how='left', sort=False)
# test_df = pd.merge(test_df, count_Buys, on=['CustomerIdx','IsinIdx'], how='left', sort=False)
# valid_df = pd.merge(valid_df, count_Buys, on=['CustomerIdx','IsinIdx'], how='left', sort=False)
# count_Sells = train_df.groupby(['CustomerIdx','IsinIdx'])['Sell'].sum().reset_index()
# count_Sells.columns = ['CustomerIdx','IsinIdx','Sells_Count_UI']
# train_df = pd.merge(train_df, count_Sells, on=['CustomerIdx','IsinIdx'], how='left', sort=False)
# test_df = pd.merge(test_df, count_Sells, on=['CustomerIdx','IsinIdx'], how='left', sort=False)
# valid_df = pd.merge(valid_df, count_Sells, on=['CustomerIdx','IsinIdx'], how='left', sort=False)
# =============================================================================


# --> Reshape Train <--
countByCustomer = train_df.groupby(["CustomerIdx","IsinIdx"])["CustomerInterest"].count().reset_index()
countByCustomer.columns = ["CustomerIdx","IsinIdx","CountByCustomer"]
train_df = pd.merge(train_df, countByCustomer, on=['CustomerIdx','IsinIdx'], how='left', sort=False)
train_df = train_df[train_df['CountByCustomer'] > 1]
del train_df['CountByCustomer']

# --> Preparing for Train <--
idx_columns = ['CustomerIdx','IsinIdx','TickerIdx']
for column in idx_columns:
    train_df.drop(column, axis=1, inplace=True)
    valid_df.drop(column, axis=1, inplace=True)
    test_df.drop(column, axis=1, inplace=True)

y_train = train_df['CustomerInterest']
y_valid = valid_df['CustomerInterest']
del test_df['DateKey'], test_df['CustomerInterest']
del train_df['TradeDateKey'], train_df['CustomerInterest']
del valid_df['TradeDateKey'], valid_df['CustomerInterest']


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
    'nthread':7,
    'random_state': 99, 
    'silent': True}
    
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#watchlist = [(d_train, 'train')]
model = xgb.train(params, d_train, 500, watchlist, 
                  maximize=False, early_stopping_rounds = 5, 
                  verbose_eval=1)

xgb.plot_importance(model)

#preds = model.predict(xgb.DMatrix(test_df))
#submission['CustomerInterest'] = preds

#submission.to_csv(path2+"version_014.csv", index=False)

######

