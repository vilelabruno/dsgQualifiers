############################
# Created by: Joao Marreta #
############################

import pandas as pd
import numpy as np
from tqdm import tqdm

class NormalEncoding(object):

	def __init__(self, C=10):
		self.C = C

	def fit(self, data, y, features='all'):
		self.y = y
		if features == 'all':
			self.features = sorted([i for i in data.columns if data[i].dtype == 'O'])
		else: 
			self.features = features

		self.values = dict()
		data["target"] = y
		for feature in self.features:
			groupby_feature = data.groupby([feature])
			feat_df = groupby_feature.target.mean().reset_index()
			feat_df.columns=[feature, "NME_%s" % feature]
			self.values[feature] = feat_df
		data.drop(["target"], axis=1, inplace=True)
		return self.values
    		
	def transform(self, data):
		features = [i for i in self.values if i in data.columns]
		for feature in tqdm(features, desc="merging"):
			data = pd.merge(data, self.values[feature], how="left", on=feature, sort=False)
		return data

	def fit_transform(self, data, y, features='all'):
		self.fit(data, y, features)              
		return self.transform(data)