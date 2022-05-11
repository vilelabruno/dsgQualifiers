import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
path = "../../input/"
path2 = "../../output/"
mm = pd.read_csv(path+"MarketData_Macro.csv")

mm = mm.rename(columns={'FX_USD.ARS': 'ARS'})
mm = mm.rename(columns={'FX_USD.AUD': 'AUD'})
mm = mm.rename(columns={'FX_USD.BRL': 'BRL'})
mm = mm.rename(columns={'FX_USD.CAD': 'CAD'})
mm = mm.rename(columns={'FX_USD.CHF': 'CHF'})
mm = mm.rename(columns={'FX_USD.CNO': 'CNO'})
mm = mm.rename(columns={'FX_USD.CNY': 'CNY'})
mm = mm.rename(columns={'FX_USD.EUR': 'EUR'})
mm = mm.rename(columns={'FX_USD.GBP': 'GBP'})
mm = mm.rename(columns={'FX_USD.HKD': 'HKD'})
mm = mm.rename(columns={'FX_USD.IDR': 'IDR'})
mm = mm.rename(columns={'FX_USD.JPY': 'JPY'})
mm = mm.rename(columns={'FX_USD.NOK': 'NOK'})
mm = mm.rename(columns={'FX_USD.SGD': 'SGD'})
mm = mm.rename(columns={'FX_USD.TRY': 'TRY'})
mm = mm.rename(columns={'FX_USD.ZAR': 'ZAR'})

for col in mm.columns:
	if (col != 'ARS') & (col != 'DateKey') & (col != 'AUD') & (col != 'BRL') & (col != 'CAD') & (col != 'CHF') & (col != 'CNO') & (col != 'CNY') & (col != 'EUR') & (col != 'GBP') & (col != 'HKD') & (col != 'IDR') & (col != 'JPY') & (col != 'NOK') & (col != 'SGD') & (col != 'TRY') & (col != 'ZAR'):
		mm.drop(col, axis=1, inplace=True)


mm.to_csv(path+"mmOnlyConversion.csv", index=False)
