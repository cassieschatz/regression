#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:42:57 2021

@author: cassieschatz
"""
#Import packages
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def printPlot(var, x, y, m, b):
    plt.title("Relationship between Salary and " + var)
    m = linReg.coef_
    b = linReg.intercept_
    plt.xlabel("Salary, Formula: s(x) = " + str(m) + "x + " + str(b))
    plt.ylabel(var)
    plt.scatter(x,y, color='red')
    plt.plot(x, linReg.predict(x),color='blue')
    plt.show()
    plt.clf()
    
def MSE(actual, predict):
    errors = []
    #Find the individual errors
    for i in range(0,len(actual)):
        diff = actual[i] - predict[i]
        diffSq = diff * diff
        errors.append(diffSq)
    
    #Take the mean:
    sumVal = 0
    for i in range(0, len(errors)):
        sumVal = sumVal + errors[i]
    
    return (sumVal/len(errors));
    
    

#Get data
raw_data = pd.read_csv('baseball-9192.csv');

labels = ["BattingAvg", "OnBasePct", "Runs", "Hits", "Doubles", "Triples", "HomeRuns", "RBI", "Walks", "Strikeouts", "StolenBases", "Errors", "FreeAgencyElig", "FreeAgent9192", "ArbitrationElig", "Arbitration9192"]
labels2 = ["BattingAvg", "OnBasePct", "Runs", "Hits", "Doubles", "Triples", "HomeRuns", "RBI", "Walks", "Strikeouts", "StolenBases", "Errors", "FreeAgencyElig", "FreeAgent9192", "ArbitrationElig", "Arbitration9192"]

errors = []
for var in labels:
    #Initialize the varables
    x = raw_data["Salary"];
    x = x.values.reshape(len(x),1)
    y = raw_data[var]
    y = y.values.reshape(len(y),1)
    scaler = StandardScaler()
    scaler.fit(y)
    scaled_features = scaler.transform(y)
    scaled_data = pd.DataFrame(y)
    
    #Find the regression
    linReg = LinearRegression().fit(x,y)
    
    #Find the error (maybe use a diferent error metric?)
    #r_sq = np.mean((y - linReg.predict(x))**2)
    r_sq = MSE(y, linReg.predict(x))
    #Showcase your findings to the world
    errors.append(r_sq)
    
    printPlot(var, x, y, linReg.coef_, linReg.intercept_)
    if(linReg.coef_ < 0.001):
        errors.remove(r_sq);
        labels2.remove(var);

#Finding the smallest error
index = 0
smallest = errors[index]
for i in range(0,len(labels2)):
    print(labels2[i] + " " + str(errors[i]))
    if(smallest > errors[i]):
        index = i
        smallest = errors[i]

print("The smallest error exists when the y is " + labels[index])
    