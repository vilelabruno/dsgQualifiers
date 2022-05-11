import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

path = "/home/bruno/dsg_2018/input/"
path2 = "/home/bruno/dsg_2018/output/"

pred1 = pd.read_csv(path2+"v_h_005_2.csv")		#0.69960
pred2 = pd.read_csv(path2+"humberto2.csv")		#0.68310
pred3 = pd.read_csv(path2+"v_007.csv")		    #0.74380
pred4 = pd.read_csv(path2+"v_h_006.csv")		#0.74835

aux_pred1 = pred1.sort_values(by=['CustomerInterest'])
aux_pred2 = pred2.sort_values(by=['CustomerInterest'])
aux_pred3 = pred3.sort_values(by=['CustomerInterest'])
aux_pred4 = pred4.sort_values(by=['CustomerInterest'])

aux_pred1 = pred1
aux_pred2 = pred2
aux_pred3 = pred3
aux_pred4 = pred4

