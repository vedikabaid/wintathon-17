
#importing required libraries
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


train = pd.read_csv("traindata.csv")
test =  pd.read_csv("testdata.csv")


features_trains = train[["Ranking"]].values
target_trains = train[["Label"]].values

features_tests = test[["Ranking"]].values
target_tests= test[["Label"]].values

#Using logistic regression :)
clf = LogisticRegression()
clf.fit(features_trains, target_trains)
#predicting performance for logistic regression
print 'score- logistic regression ', clf.score(features_tests, target_tests)


#Using random forest classifier :))
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
forest = forest.fit(features_trains,target_trains)
#predicting performance for random forest regression
print 'score -random forest' , forest.score(features_tests, target_tests)