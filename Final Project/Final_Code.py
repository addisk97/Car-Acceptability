# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 21:36:56 2019

@author: addis_000
Addisalem Kebede
CptS 315 Final Project
"""

#using sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn import metrics
import csv



def digestData():
    carData = open("car.data", "r")
    
    data = []
    classes = []

    line = carData.readline()
    while(line):
        split = line.split(",")
        
        features = split[0:6]
        class_ = split[-1]
        class_ = class_[0:-1]
        
        numFeatures = featureToInteger(features)
        classNum = classToInteger(class_)
        
        classes.append(classNum)
        data.append(numFeatures)
        
        line = carData.readline()
        
    return data, classes

def breakUpData(data, classes, testSize):
    dataForTraining, dataForTesting, classesForTraining, classesForTesting = train_test_split(data,classes, test_size = testSize)
    return dataForTraining, dataForTesting, classesForTraining, classesForTesting

def doNaiveBayes(dataForTraining, dataForTesting, classesForTraining):
    cars = GaussianNB()
    cars.fit(dataForTraining,classesForTraining)
    prediction = cars.predict(dataForTesting)
    return prediction

def doPerceptron(dataForTraining, dataForTesting, classesForTraining, iterations, learningRate, v):
    ppn = Perceptron(max_iter = iterations, eta0 = learningRate, verbose = v)
    ppn.fit(dataForTraining,classesForTraining)
    prediction = ppn.predict(dataForTesting)
    return prediction

def determineAccuracy(algorithmName, actualValues, testPredictions, testSize):
    accuracy = metrics.accuracy_score(actualValues, testPredictions)
    print("The accuracy of", algorithmName, "on the test data with test size being", testSize ,"is: ", accuracy)
    return accuracy

def featureToInteger(features):
    #changing the 1st feature to a number representation
    numFeatures = []
    if(features[0] == "vhigh"):
       num = 0
    elif(features[0] == "high"):
       num = 1
    elif (features[0] == "med"):
       num = 2
    elif (features[0] == "low"):
       num = 3
    numFeatures.append(num)
    # changing the 2nd feature to a number representation
    if(features[1] == "vhigh"):
       num = 0
    elif(features[1] == "high"):
       num = 1
    elif (features[1] == "med"):
       num = 2
    elif (features[1] == "low"):
       num = 3
    numFeatures.append(num)
    # changing the 3rd feature to a number representation
    if(features[2] == "2"):
       num = 0
    elif(features[2] == "3"):
       num = 1
    elif (features[2] == "4"):
       num = 2
    elif (features[2] == "5more"):
       num = 3
    numFeatures.append(num)
    
    # changing the 4th feature to a number representation
    if (features[3] == "2"):
       num = 0
    elif (features[3] == "4"):
       num = 1
    elif (features[3] == "more"):
       num = 2
    numFeatures.append(num)
    # changing the 5th feature to a number representation
    if (features[4] == "small"):
       num = 0
    elif (features[4] == "med"):
       num = 1
    elif (features[4] == "big"):
       num = 2
    numFeatures.append(num)
    #changing the 6th feature to a number representation
    if (features[5] == "low"):
       num = 0
    elif (features[5] == "med"):
       num = 1
    elif (features[5] == "high"):
       num = 2
    numFeatures.append(num)
    return numFeatures

def classToInteger(class_):
    classNum = 0
    #chainging the class to an integer representation
    if(class_ == "unacc"):
       classNum = 0
    elif(class_ == "acc"):
       classNum = 1
    elif (class_ == "good"):
       classNum = 2
    elif (class_ == "vgood"):
       classNum = 3
    return classNum

#output
def outputData(accuracy, testSize, algorithm):
    #Want the different accuracy scores along with the data split used to get them
    if(algorithm == "Perceptron"):
        with open('accuracy_results_Perceptron.csv', mode = 'a') as results:
            results_writer = csv.writer(results, delimiter=',')
            results_writer.writerow([accuracy, testSize])
    elif(algorithm == "Naive Bayes"):
        with open('accuracy_results_NB.csv', mode = 'a') as results:
            results_writer = csv.writer(results, delimiter=',')
            results_writer.writerow([accuracy, testSize])

#getting the Perceptron accuracy for sample splits of (train-test): 25-75 ,50-50, 75-25, 95-5
def getPerceptronAccuracy():
    iterations = 5

    print("\n**************************** PERCEPTRON FOR 25-75 ****************************")
    count = 0
    while(count < iterations):
        data, classes = digestData()
        dataForTraining, dataForTesting, classesForTraining, classesForTesting = breakUpData(data, classes, .75)
        testPredictions = doPerceptron(dataForTraining, dataForTesting, classesForTraining, 50, 1,0)
        accuracy = determineAccuracy("Perceptron", classesForTesting, testPredictions, .75)
        outputData(accuracy, .75, "Perceptron")
        count = count + 1

    print("\n**************************** PERCEPTRON FOR 50-50 ****************************")
    count = 0
    while(count < iterations):
        data, classes = digestData()
        dataForTraining, dataForTesting, classesForTraining, classesForTesting = breakUpData(data, classes, .50)
        testPredictions = doPerceptron(dataForTraining, dataForTesting, classesForTraining, 50, 1,0)
        accuracy = determineAccuracy("Perceptron", classesForTesting, testPredictions, .50)
        outputData(accuracy, .50, "Perceptron")
        count = count + 1

    print("\n**************************** PERCEPTRON FOR 75-25 ****************************")
    count = 0
    while(count < iterations):
        data, classes = digestData()
        dataForTraining, dataForTesting, classesForTraining, classesForTesting = breakUpData(data, classes, .25)
        testPredictions = doPerceptron(dataForTraining, dataForTesting, classesForTraining, 50, 1,0)
        accuracy = determineAccuracy("Perceptron", classesForTesting, testPredictions, .25)
        outputData(accuracy, .25, "Perceptron")
        count = count + 1

    print("\n**************************** PERCEPTRON FOR 95-5 ****************************")
    count = 0
    while(count < iterations):
        data, classes = digestData()
        dataForTraining, dataForTesting, classesForTraining, classesForTesting = breakUpData(data, classes, .05)
        testPredictions = doPerceptron(dataForTraining, dataForTesting, classesForTraining, 50, 1,0)
        accuracy = determineAccuracy("Perceptron", classesForTesting, testPredictions, .05)    
        outputData(accuracy, .05, "Perceptron")
        count = count + 1
    
#getting the NB accuracy for sample splits of (train-test): 25-75 ,50-50, 75-25, 95-5
def getNaiveBayesAccuracy():

    iterations = 5
    count = 0
    print("\n**************************** NAIVE BAYES FOR 25-75 ****************************")
    while(count < iterations):
        data, classes = digestData()
        dataForTraining, dataForTesting, classesForTraining, classesForTesting = breakUpData(data, classes, .75)
        testPredictions = doNaiveBayes(dataForTraining, dataForTesting, classesForTraining)
        accuracy = determineAccuracy("Naive Bayes", classesForTesting, testPredictions, .75)
        outputData(accuracy, .75, "Naive Bayes")
        count = count + 1 
        
    print("\n**************************** NAIVE BAYES FOR 50-50 ****************************")
    count = 0
    while(count < iterations):
        data, classes = digestData()
        dataForTraining, dataForTesting, classesForTraining, classesForTesting = breakUpData(data, classes, .50)
        testPredictions = doNaiveBayes(dataForTraining, dataForTesting, classesForTraining)
        accuracy = determineAccuracy("Naive Bayes", classesForTesting, testPredictions, .50)
        outputData(accuracy, .50, "Naive Bayes")
        count = count + 1

    print("\n**************************** NAIVE BAYES FOR 75-25 ****************************") 
    count = 0
    while(count < iterations):
        data, classes = digestData()
        dataForTraining, dataForTesting, classesForTraining, classesForTesting = breakUpData(data, classes, .25)
        testPredictions = doNaiveBayes(dataForTraining, dataForTesting, classesForTraining)
        accuracy = determineAccuracy("Naive Bayes", classesForTesting, testPredictions, .25)
        outputData(accuracy, .25, "Naive Bayes")
        count = count + 1
        
    print("\n**************************** NAIVE BAYES FOR 95-5 ****************************")
    count = 0
    while(count < iterations):
        data, classes = digestData()
        dataForTraining, dataForTesting, classesForTraining, classesForTesting = breakUpData(data, classes, .05)
        testPredictions = doNaiveBayes(dataForTraining, dataForTesting, classesForTraining)
        accuracy = determineAccuracy("Naive Bayes", classesForTesting, testPredictions, .05)
        outputData(accuracy, .05, "Naive Bayes")
        count = count + 1

def main():
    with open('accuracy_results_Perceptron.csv', mode='w') as results:
        results_writer = csv.writer(results, delimiter=',')
        results_writer.writerow(["Accuracy", "TestSize"])

    with open('accuracy_results_NB.csv', mode='w') as results:
        results_writer = csv.writer(results, delimiter=',')
        results_writer.writerow(["Accuracy", "TestSize"])

    getPerceptronAccuracy()
    getNaiveBayesAccuracy()

if __name__ == '__main__':
    main()
    
