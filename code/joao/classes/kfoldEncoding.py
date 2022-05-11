############################
# Created by: Joao Marreta #
############################

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from normalEncoding import NormalEncoding

class KfoldEncoding(object):
	
	def __init__(self, groups=5, seed=99):
		self.groups = groups
		self.seed = seed

	def fit(self, data, y, features='all'):
		self.y = y
		if features == 'all':
			self.features = sorted([i for i in data.columns if data[i].dtype == 'O'])
		else: 
			self.features = features

		self.values = dict()
		NE = NormalEncoding()
		for feature in self.features:
			aux_df = pd.DataFrame()
			folds = KFold(n_splits=self.groups, shuffle=True, random_state=self.seed)
			for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data)):
				new_data = data.copy()
				data_y = self.y.iloc[trn_idx]
				new_data = NE.fit_transform(data=new_data, y=data_y, features=[feature])
				data_y = new_data.iloc[val_idx]
				aux_df = pd.concat([aux_df, data_y])
			return_feats=['NME_'+feature]
			self.values[feature] = aux_df[return_feats]
			self.values[feature].columns = ['KFE_'+feature]
		return self.values

	def transform(self, data):
		features = [i for i in self.values if i in data.columns]
		for feature in tqdm(features, desc="merging"):
			data["KFE_"+feature] = self.values[feature]
		return data

	def fit_transform(self, data, y, features='all'):
		self.fit(data, y, features)              
		return self.transform(data)