#%% Importando as bibliotecas

import pandas as pd
import numpy as np
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt

#%% Lendo as bases

path = '/Users/Felipe Ferreira/Dropbox/dsg_2018/input/'

test = pd.read_csv(path + 'Challenge_20180423.csv')
trade = pd.read_csv(path + 'Trade.csv')
isin = pd.read_csv(path + 'Isin.csv')
customer = pd.read_csv(path + 'Customer.csv')
market = pd.read_csv(path + 'Market.csv')
mdm = pd.read_csv(path + 'MarketData_Macro.csv')

#%% Dando merge nas bases

main = test.merge(isin, how='left', left_on='IsinIdx', right_on='IsinIdx')