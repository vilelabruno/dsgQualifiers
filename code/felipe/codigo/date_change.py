# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:29:56 2018

@author: Felipe Ferreira
"""

#%% Importando as bibliotecas

import pandas as pd
import numpy as np
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt


#%% Lendo as bases

path = '/Users/Felipe Ferreira/Dropbox/dsg_2018/input/'

isin = pd.read_csv(path + 'Isin.csv')

#%% Função que modifica data

def monthDays(actual, isBissexto):
    daysPassed = 0
    daysOfMonth = [31,28,31,30,31,30,31,31,30,31,30,31]
    if isBissexto:
        daysOfMonth[1] = 29
    for x in range(0, actual-1):
        daysPassed = daysPassed + daysOfMonth[x]
    return daysPassed;

#%%
def changeToDays(date):
    bissexto = False
    year = int(date/10000)
    date = date - year*10000
    month = int(date/100)
    date = date - month*100
    year = year - 2016
    if year%4==0 and year%100!=0 or year%400==0:
        bissexto = True
    else:
        date = date + round(year/4 + 0.5)
    date = (year * 365) + date + monthDays(month, bissexto)
    return date;
    
#%%
def changeTime(myList):
    newList = [changeToDays(x) for x in myList]
    return newList;

#%%
isin['daysToMature'] = changeTime(isin['ActualMaturityDateKey'])