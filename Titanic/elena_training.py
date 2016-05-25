# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 23:00:00 2015

@author: elena cuoco 2
"""
#Titatic competitor usign pandas and scikit library
import numpy as np
import pandas as pd
from pandas import  DataFrame
from patsy import dmatrices
import string
from operator import itemgetter
#json library for settings file
import json
# import the machine learning library that holds the randomforest
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split,StratifiedShuffleSplit,StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import classification_report
#joblib library for serialization
from sklearn.externals import joblib
from utils import clean_and_munge_data,report

##Read configuration parameters

train_file='input/train.csv'
MODEL_PATH='model/'
test_file='input/test.csv'
SUBMISSION_PATH='submission/'
seed= 50

print train_file,seed

#read data
traindf=pd.read_csv(train_file)
##clean data
df=clean_and_munge_data(traindf)
########################################formula################################
 
formula_ml='Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size' 

y_train, x_train = dmatrices(formula_ml, data=df, return_type='dataframe')
y_train = np.asarray(y_train).ravel()
print y_train.shape,x_train.shape

##select a train and test set
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2,random_state=seed)
#instantiate and fit our model

clf=RandomForestClassifier(n_estimators=500,
                           criterion='entropy',
                           max_depth=5,
                           min_samples_split=1,
                           min_samples_leaf=1,
                           max_features='auto',
                           bootstrap=False,
                           oob_score=False,
                           n_jobs=1,
                           random_state=seed,
                           verbose=0,
                           min_density=None,
                           compute_importances=None)

###compute grid search to find best paramters for pipeline
param_grid = dict( )
##classify pipeline
pipeline=Pipeline([ ('clf',clf) ])
grid_search = GridSearchCV(pipeline,
    param_grid=param_grid,
    verbose=3,
    scoring='average_precision',
    cv=StratifiedShuffleSplit(Y_train,
        n_iter=10,
        test_size=0.2,
        train_size=None,
        indices=None, 
        random_state=seed,
        n_iterations=None)).fit(X_train, Y_train)

# Score the results
###print result
print("Best score: %0.3f" % grid_search.best_score_)
print(grid_search.best_estimator_)
report(grid_search.grid_scores_)
 
print('-----grid search end------------')
print ('on all train set')
scores = cross_val_score(grid_search.best_estimator_,
    x_train,
    y_train,
    cv=3,
    scoring='accuracy')

print scores.mean(),scores

print ('on test set')
scores = cross_val_score(grid_search.best_estimator_,
    X_test,
    Y_test,
    cv=3,
    scoring='accuracy')

print scores.mean(),scores

# Score the results
print(classification_report(Y_train, grid_search.best_estimator_.predict(X_train) ))
print('test data')
print(classification_report(Y_test, grid_search.best_estimator_.predict(X_test) ))

#serialize training
model_file=MODEL_PATH+'model-rf.pkl'
joblib.dump(grid_search.best_estimator_, model_file)
