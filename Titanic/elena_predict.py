# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 23:05:02 2015

@author: admin
"""

#Titatic competitor usign pandas and scikit library
import numpy as np
import pandas as pd
from pandas import  DataFrame
from patsy import dmatrices

#json library for settings file
import json
# import the machine learning library that holds the randomforest
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split,StratifiedShuffleSplit,StratifiedKFold
from sklearn import preprocessing
#joblib library for serialization
from sklearn.externals import joblib

from utils import clean_and_munge_data,report
#Read data and configuration parameters#


train_file='intput/train.csv'
MODEL_PATH='model/'
test_file='input/test.csv'
SUBMISSION_PATH='submission/'
seed= 50

print test_file, seed

#read data
testdf=pd.read_csv(test_file)

ID=testdf['PassengerId']
##clean data
df_test=clean_and_munge_data(testdf)
df_test['Survived'] =  [0 for x in range(len(df_test))]

print df_test.shape
########################################formula################################
 
formula_ml='Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size' 

y_p,x_test = dmatrices(formula_ml, data=df_test, return_type='dataframe')
y_p = np.asarray(y_p).ravel()
print  y_p.shape,x_test.shape
#serialize training
model_file=MODEL_PATH+'model-rf.pkl'
clf = joblib.load(model_file)
####estimate prediction on test data set
y_p=clf.predict(x_test).astype(int)
print y_p.shape

outfile=SUBMISSION_PATH+'prediction-BS.csv'
dfjo = DataFrame(dict(Survived=y_p,PassengerId=ID), columns=['Survived','PassengerId'])
dfjo.to_csv(outfile,index_label=None,index_col=False,index=False)


#