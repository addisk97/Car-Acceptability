# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:16:33 2019

@author: addis_000
"""

from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron


X,y = load_digits(return_X_y=True)

ppn = Perceptron(max_iter = 15, verbose = 1)

ppn.fit(X,y)

ppn.score(X,y)




def doPerceptron(dataTrain, dataTest, classTrain, classTest, iterations, learningRate):
    ppn = Perceptron(max_iter = 15, verbose = 1)
    ppn.fit(data,classes)
    


data, classes = digestData()
xTrain, xTest, yTrain, yTest = breakUpData(data, classes)




