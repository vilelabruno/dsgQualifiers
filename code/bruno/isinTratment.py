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
isin_df = pd.read_csv(path+"Isin.csv")

#categorical_feats = [f for f in isin_df.columns if isin_df[f].dtype == 'object']
#all_cat = categorical_feats
#for i in categorical_feats:
#    if (len(isin_df[i].unique()) > 00):
#        isin_df[i], indexer = pd.factorize(isin_df[i])
#    else:
#        ohe = pd.get_dummies(isin_df[i])
#        columns = ohe.columns
#        for j in columns:
#            isin_df[j] = ohe[j]
#        isin_df.drop(i, axis=1, inplace=True)
#categorical_feats = [f for f in isin_df.columns if isin_df[f].dtype == 'object']
#print(categorical_feats)