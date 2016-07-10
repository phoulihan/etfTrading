# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 08:55:19 2016

@author: xilin
"""
import pandas as pd
import datetime
import numpy as np
import pandas.io.data as web
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib import style
import statsmodels.formula.api as lm
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
from decimal import Decimal
from pymongo import MongoClient
from sklearn import linear_model, datasets
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn import linear_model, datasets
import imp
from sklearn.naive_bayes import GaussianNB
from pymongo import MongoClient
import collections, re

mongo = MongoClient('127.0.0.1', 27017)
mongoDb = mongo['priceData']
mongoColl = mongoDb['crspData']

style.use('ggplot')

start = datetime.datetime(2010,1,1)
end = datetime.datetime(2015,1,1)

statSig = .05
postThresh = .8
theWindow = 30

theTickers = np.sort(np.array(mongoColl.distinct('ticker')))
theTickers = [s.strip('$') for s in theTickers]
numTickers = len(theTickers)

tempBase = web.DataReader("XOM","yahoo",start,end)
tempBase['retBase'] = np.log(tempBase['Adj Close'].astype(float)) - np.log(tempBase['Adj Close'].astype(float).shift(1))
tempBase.reindex()

thePerf = list()
for i in range(0,10):
    print(theTickers[i])
    try:
        temp = web.DataReader(theTickers[i],"yahoo",start,end)
        temp['ret'] = np.log(temp['Adj Close'].astype(float)) - np.log(temp['Adj Close'].astype(float).shift(1))
        temp.reindex()
        
        tempData = pd.merge(tempBase,temp,how='outer', left_index=True, right_index=True)
        tempData = tempData[['retBase','ret']]
        
        tempData = tempData.dropna()
        theLen = len(tempData)
        testSize = .10
        
        tempData['theDiff'] = tempData['retBase'] - tempData['ret']
        
        theCor = pearsonr(tempData['retBase'],tempData['ret'])
        rollRet = float(0)
        
        if(theCor[1] <= statSig):
            tempData['rollCor'] = pd.rolling_corr(tempData['retBase'],tempData['ret'],theWindow) #rollCorrelation
            
            tempData = tempData.dropna()
            
            testLen = int(round(testSize*theLen,0))
            trainLen = int(theLen  - testLen)
            try:
                y = tempData['ret'][1:theLen] #next day assset return
                X = tempData[['retBase','theDiff','rollCor']][0:theLen-1] #event day features
            
                trainY = y[0:(trainLen-1)]
                testY = y[trainLen:theLen]
                
                trainX = X[0:(trainLen-1)]
                testX = X[trainLen:theLen]
            
                model = RandomForestClassifier(n_estimators=25,random_state=42)
                model.fit(trainX,np.sign(trainY.astype(float)))
                postProbs = model.predict_proba(testX)
                theClasses = model.classes_ #[-1.  0.  1.]
                neg = int(np.where(theClasses == -1.0)[0])
                pos = int(np.where(theClasses == 1.0)[0])
            
                theLongs = postProbs[:,pos][postProbs[:,pos] >= postThresh] #LONG POSITIONS
                theShorts = postProbs[:,neg][postProbs[:,neg] >= postThresh] #SHORT POSITIONS
                
                corLong = np.sign(theLongs) == 1
                incLong = np.sign(theLongs) != 1
                longRet = float(np.sum(testY[corLong])) - float(np.sum(testY[incLong])) 
                
                corShort = np.sign(theShorts) == -1
                incShort = np.sign(theShorts) != -1
                shortRet = -float(np.sum(testY[corShort])) + float(np.sum(testY[incShort])) 
                
                theRet = float(longRet) + float(shortRet)
                rollRet = float(rollRet) + float(theRet)
                thePerf.append(theRet)
                print("Overall Return: " + str(theRet))
            except Exception, (e):
                print(e)         
            pass
    except Exception, (e):
        print(e)         
    pass      

print("Sharpe: " + str(np.mean(thePerf)/np.std(thePerf)))
temp = np.where(np.asarray(thePerf) < 0)
print("Sortino: " + str(np.mean(thePerf)/np.std(np.asarray(thePerf)[temp])))
#http://pandas.pydata.org/pandas-docs/stable/remote_data.html