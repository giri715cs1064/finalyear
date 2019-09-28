# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:18:55 2019

@author: giridhar
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv(r'C:\Users\giridhar\Desktop\sample.csv')

x = dataset.iloc[:,1:19].values
y= dataset.iloc[:,19].values

#print(x)
#print(y)

#from sklearn.cross_validation import train_test_split
#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
perm = np.random.permutation(len(x))



#x_train, x_test = x[:,1:100,565:620],x[:,101:125,620:650]
#y_train, y_test = y[:,1:100,565:620],y[:,101:125,620:650]



x_train, x_test = x[perm][100:],x[perm][:100]
y_train, y_test = y[perm][100:],y[perm][:100]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(x_train,y_train)
y_pred = mlp.predict(x_test)

from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))