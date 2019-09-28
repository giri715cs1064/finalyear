# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 18:35:22 2019

@author: giridhar
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv(r'C:\Users\giridhar\Desktop\project\data2.csv')

x = dataset.iloc[:,1:4558].values
y= dataset.iloc[:,4558].values


from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model, 3500)
fit = rfe.fit(x, y)
print(rfe.support_)
print(rfe.ranking_)