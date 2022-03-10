#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################

#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
from sklearn import svm
from sklearn import metrics

#for lC in [10,100,1000,10000]:
lC = 10000
clf = svm.SVC(kernel='rbf', C=lC, gamma='auto')
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
print(lC)
print(metrics.accuracy_score(labels_test,pred) )
print(clf.score(features_test,labels_test))
print(sum(pred))