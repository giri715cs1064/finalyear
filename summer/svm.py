# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:54:42 2019

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

x_train, x_test = x[perm][200:],x[perm][:200]
y_train, y_test = y[perm][200:],y[perm][:200]

#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)
print(y_pred)


from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))