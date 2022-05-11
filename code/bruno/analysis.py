import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

path = "../../input/"
path2 = "../../output/"

test_df = pd.read_csv(path+"Challenge_20180423.csv")
train_df = pd.read_csv(path+"Trade.csv")
#train_df.drop('Day', axis=1, inplace=True)

aux = train_df.groupby(['CustomerIdx', 'IsinIdx']).agg({'TradeDateKey': np.min})
aux = aux.reset_index()
aux = aux.rename(columns={'TradeDateKey': 'dateMin'})
aux2 = train_df.groupby(['CustomerIdx', 'IsinIdx']).agg({'TradeDateKey': np.max})
aux2 = aux2.reset_index()
aux2 = aux2.rename(columns={'TradeDateKey': 'dateMax'})
aux = pd.merge(aux, aux2, on=['CustomerIdx', 'IsinIdx'], how='inner')
aux.to_csv(path+"dateMinMax.csv", index=False)