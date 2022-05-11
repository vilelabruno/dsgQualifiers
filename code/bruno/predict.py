import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import gc

cst = pd.read_csv('../../input/Customer.csv')

gc.collect()

print cst.Country.describe()
print('Go to dummies')
cstSector = pd.get_dummies(cst.Sector, prefix='sector_')
cstRegion = pd.get_dummies(cst.Region, prefix='region_')

cst = pd.concat([cst, cstSector, cstRegion], axis=1)
cst.drop('Sector', axis=1, inplace=True)
cst.drop('Region', axis=1, inplace=True)

subsector_counts = cst.groupby('Subsector')['CustomerIdx'].count().reset_index()
subsector_counts.columns = ['Subsector', 'count_subsector']
cst = pd.merge(cst, subsector_counts, on='Subsector', how='left', sort=False)
cst.drop('Subsector', axis=1, inplace=True)

Country_counts = cst.groupby('Country')['CustomerIdx'].count().reset_index()
Country_counts.columns = ['Country', 'count_Country']
cst = pd.merge(cst, Country_counts, on='Country', how='left', sort=False)
cst.drop('Country', axis=1, inplace=True)

print cst

del cstSector, cstRegion, subsector_counts, Country_counts
gc.collect()

isin = pd.read_csv('../../input/Isin.csv')

print isin.CouponType.describe()
isinActG = pd.get_dummies(isin.ActivityGroup, prefix='actg_')
isinSen = pd.get_dummies(isin.Seniority, prefix='sen_')
isinCur = pd.get_dummies(isin.Currency, prefix='cur_')
isinReg = pd.get_dummies(isin.Region, prefix='reg_')
isinAct = pd.get_dummies(isin.Activity, prefix='act_')
isinRiskC = pd.get_dummies(isin.RiskCaptain, prefix='riskC_')
isinCmpR = pd.get_dummies(isin.CompositeRating, prefix='cmpR_')
isinIndS = pd.get_dummies(isin.IndustrySector, prefix='indS_')
isinMktS = pd.get_dummies(isin.MarketIssue, prefix='mktS_')
isinCupT = pd.get_dummies(isin.CouponType, prefix='cupT_')

isin = pd.concat([isin, isinActG, isinSen, isinCur, isinReg, isinAct, isinRiskC, isinCmpR, isinIndS, isinMktS, isinCupT], axis=1)
isin.drop('ActivityGroup', axis=1, inplace=True)
isin.drop('Seniority', axis=1, inplace=True)
isin.drop('Currency', axis=1, inplace=True)
isin.drop('Region', axis=1, inplace=True)
isin.drop('Activity', axis=1, inplace=True)
isin.drop('RiskCaptain', axis=1, inplace=True)
isin.drop('CompositeRating', axis=1, inplace=True)
isin.drop('IndustrySector', axis=1, inplace=True)
isin.drop('MarketIssue', axis=1, inplace=True)
isin.drop('CouponType', axis=1, inplace=True)
isin.drop('ActualMaturityDateKey', axis=1, inplace=True)
isin.drop('IssueDateKey', axis=1, inplace=True)

countOwner = isin.groupby('Owner')['IsinIdx'].count().reset_index()
countOwner.columns = ['Owner', 'countOwner']
isin = pd.merge(isin, countOwner, on='Owner', how='left', sort=False)
isin.drop('Owner', axis=1, inplace=True)

countIndustrySubgroup = isin.groupby('IndustrySubgroup')['IsinIdx'].count().reset_index()
countIndustrySubgroup.columns = ['IndustrySubgroup', 'countIndustrySubgroup']
isin = pd.merge(isin, countIndustrySubgroup, on='IndustrySubgroup', how='left', sort=False)
isin.drop('IndustrySubgroup', axis=1, inplace=True)

del isinActG, isinSen, isinCur, isinReg, isinAct, isinRiskC, isinCmpR, isinIndS, isinMktS, isinCupT, countOwner, countIndustrySubgroup
gc.collect()

test = pd.read_csv('../../input/Challenge_20180423.csv')
mkt = pd.read_csv('../../input/Market.csv')
mktMacro = pd.read_csv('../../input/MarketData_Macro.csv')
data = pd.read_csv('../../input/Trade.csv')
sub = pd.read_csv('../../input/sample_submission.csv')

print('Read data and test')
#data = pd.read_csv('../input/application_train.csv')
#test = pd.read_csv('../input/application_test.csv')
#print('Shapes : ', data.shape, test.shape)
#
y = data['CustomerInterest']
del data['CustomerInterest']
#
data.drop('NotionalEUR', axis=1, inplace=True)
data.drop('Price', axis=1, inplace=True)
data.drop('TradeStatus', axis=1, inplace=True)
data.drop('TradeDateKey', axis=1, inplace=True)
test.drop('DateKey', axis=1, inplace=True)

print 'lets roll with test and train'
buysell = pd.get_dummies(data.BuySell, prefix='bs_')
data = pd.concat([data, buysell], axis=1)
del buysell
gc.collect()

buysell = pd.get_dummies(test.BuySell, prefix='bs_')
test = pd.concat([test, buysell], axis=1)
del buysell
gc.collect()
    
data = pd.merge(data, isin, how='left', on='IsinIdx')
data = pd.merge(data, cst, how='left', on='CustomerIdx')
test = pd.merge(test, isin, how='left', on='IsinIdx')
test = pd.merge(test, cst, how='left', on='CustomerIdx')
print data
#test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
#
#data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
#test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
#
#data = data.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
#test = test.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
#
#data = data.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
#test = test.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
#
#data = data.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')
#test = test.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')
#
#del avg_buro, avg_prev
gc.collect()
#
from lightgbm import LGBMClassifier
#import gc
#
gc.enable()
#
folds = KFold(n_splits=8, shuffle=True, random_state=546789)
oof_preds = np.zeros(data.shape[0])
sub_preds = np.zeros(test.shape[0])

feature_importance_df = pd.DataFrame()

feats = [f for f in data.columns if f not in ['SK_ID_CURR']]

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data)):
    trn_x, trn_y = data[feats].iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = data[feats].iloc[val_idx], y.iloc[val_idx]
    
    clf = LGBMClassifier(
        # n_estimators=1000,
        # num_leaves=20,
        # colsample_bytree=.8,
        # subsample=.8,
        # max_depth=7,
        # reg_alpha=.1,
        # reg_lambda=.1,
        # min_split_gain=.01
        n_estimators=10000,
        learning_rate=0.03,
        num_leaves = 40,
        colsample_bytree=0.9497036,
        subsample=0.8715623,
        max_depth=10,
        reg_alpha=0.041545473,
        reg_lambda=0.0735294,
        min_split_gain=0.0222415,
        min_child_weight=39.3259775,
        silent=-1,
        verbose=-1,
    )
    
    clf.fit(trn_x, trn_y, 
            eval_set= [(trn_x, trn_y), (val_x, val_y)], 
            eval_metric='auc', verbose=100, early_stopping_rounds=300  #30
           )
    
    oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += clf.predict_proba(test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feats
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()
    
print('Full AUC score %.6f' % roc_auc_score(y, oof_preds)) 

test['TARGET'] = sub_preds

test[['SK_ID_CURR', 'TARGET']].to_csv('first_submission.csv', index=False)

# Plot feature importances
#cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
#    by="importance", ascending=False)[:50].index
#
#best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
#
#plt.figure(figsize=(8,10))
#sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
#plt.title('LightGBM Features (avg over folds)')
#plt.tight_layout()
#plt.savefig('lgbm_importances.png')
#
## Plot ROC curves
#plt.figure(figsize=(6,6))
#scores = [] 
#for n_fold, (_, val_idx) in enumerate(folds.split(data)):  
#    # Plot the roc curve
#    fpr, tpr, thresholds = roc_curve(y.iloc[val_idx], oof_preds[val_idx])
#    score = roc_auc_score(y.iloc[val_idx], oof_preds[val_idx])
#    scores.append(score)
#    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))
#
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
#fpr, tpr, thresholds = roc_curve(y, oof_preds)
#score = roc_auc_score(y, oof_preds)
#plt.plot(fpr, tpr, color='b',
#         label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
#         lw=2, alpha=.8)
#
#plt.xlim([-0.05, 1.05])
#plt.ylim([-0.05, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('LightGBM ROC Curve')
#plt.legend(loc="lower right")
#plt.tight_layout()
#
#plt.savefig('roc_curve.png')
#
## Plot ROC curves
#plt.figure(figsize=(6,6))
#precision, recall, thresholds = precision_recall_curve(y, oof_preds)
#score = roc_auc_score(y, oof_preds)
#plt.plot(recall, precision, color='b',
#         label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
#         lw=2, alpha=.8)
#
#plt.xlim([-0.05, 1.05])
#plt.ylim([-0.05, 1.05])
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('LightGBM Recall / Precision')
#plt.legend(loc="lower right")
#plt.tight_layout()
#
#plt.savefig('recall_precision_curve.png')
#print chl#