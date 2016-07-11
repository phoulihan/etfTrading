# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 08:55:19 2016

@author: xilin
"""
import math
import pandas as pd
import datetime, dateutil.parser
import numpy as np
import pandas_datareader.data as web
import statsmodels.tsa.stattools as ts
from pandas_datareader import data, wb
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
import nltk
from sklearn import cross_validation
from sklearn.svm import SVC

mongo = MongoClient('127.0.0.1', 27017)
mongoDb = mongo['priceData']
mongoColl = mongoDb['crspData']

style.use('ggplot')

thePath = 'C:/Users/xilin/gitHubCode/etfTrading/'

start = datetime.datetime(2000,1,1)
end = datetime.datetime(2016,7,10)

#variables
statSig = .05 #obvious
postThresh = .8 #posterior probability threshold
corThresh = .15 #arbitrary p-vaue for pearson correl
theWindow = 10 #arbitrary rolling window
rollRet = float(0)
totalLong = 0
totalShort = 0
rollLongRight = 0
rollShortRight = 0
testSize = .90

theTickers = np.sort(np.array(mongoColl.distinct('ticker')))
theTickers = [s.strip('$') for s in theTickers]
numTickers = len(theTickers)

baseTicker = "TIP"
tempBase = web.DataReader(baseTicker,"yahoo",start,end)
tempBase['retBase'] = np.log(tempBase['Adj Close'].astype(float)) - np.log(tempBase['Adj Close'].astype(float).shift(1))
#tempBase['retBase'] = np.log(tempBase['Close'].astype(float)) - np.log(tempBase['Open'].astype(float))
tempBase = tempBase[['retBase','Close']]
tempBase.columns = ['retBase','closeBase']
tempBase.reindex()
tempBase = tempBase.dropna()

thePerf = list()
finalData = pd.DataFrame()
for i in range(0,numTickers):
    try:
        tempData = web.DataReader(theTickers[i],"yahoo",start,end)
        tempData['retClose'] = np.log(tempData['Adj Close'].astype(float)) - np.log(tempData['Adj Close'].astype(float).shift(1))
        tempData['ret'] = np.log(tempData['Close'].astype(float)) - np.log(tempData['Open'].astype(float))
        tempData = tempData.dropna()
        tempData = tempData[['retClose','ret','Close']]
        tempData.reindex()
        tempData = pd.merge(tempBase,tempData,how='outer', left_index=True, right_index=True)
        tempData['diff'] = tempData['closeBase'] - tempData['Close']
        tempData = tempData.dropna()

        tempData = tempData[['retBase','ret','retClose','diff']]
        
        theLen = len(tempData)
        
        #theCor = pearsonr(tempData['retBase'],tempData['retClose'])        
        
        gCause = ts.grangercausalitytests(tempData[['retClose','retBase']],1,verbose=False)[1][0]['params_ftest'][1] #second position --> first position

        #if(theCor[1] <= statSig and (theCor[0] >= corThresh or theCor[0] <= -corThresh)):
        if(gCause <= statSig):
            tempData['rollMean'] = tempData['ret'].rolling(window=theWindow).mean()
            tempData['rollMeanBase'] = tempData['retBase'].rolling(window=theWindow).mean()
            tempData['rollCor'] = pd.rolling_corr(tempData['retBase'],tempData['ret'],theWindow) #rollCorrelation
            tempData = tempData.dropna()
            
            trainLen = int(round(testSize*theLen,0))
            testLen = int(theLen  - trainLen)
            try:
                y = tempData['ret'][1:theLen] #next day assset return
                X = tempData[['retBase','rollCor','rollMeanBase','rollMean','diff']][0:theLen-1] #event day features
                
                trainY = y[0:trainLen]
                testY = y[trainLen:theLen]
                
                trainX = X[0:trainLen]
                testX = X[trainLen:theLen]
            
                tDate = list(trainY.index.values)
                startTrainDate = str(tDate[0])[:10]
                endTrainDate = str(tDate[len(tDate)-1])[:10]
                
                tDate = list(testY.index.values)
                startTestDate = str(tDate[0])[:10]
                endTestDate = str(tDate[len(tDate)-1])[:10]

                model = RandomForestClassifier(n_estimators=25,random_state=42)
                #model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=300,learning_rate=1,algorithm="SAMME")
                #model = linear_model.LogisticRegression(C=1e5)
                #model = SVC(kernel='rbf', class_weight=None)
                #model = GaussianNB()

                model.fit(trainX,np.sign(trainY))

                postProbs = model.predict_proba(testX)

                theClasses = model.classes_ #[-1.  0.  1.]
                neg = int(np.where(theClasses == -1.0)[0])
                pos = int(np.where(theClasses == 1.0)[0])
            
                theLongs = np.where(postProbs[:,pos] >= postThresh)[0] #LONG POSITIONS
                theShorts = np.where(postProbs[:,neg] >= postThresh)[0] #SHORT POSITIONS
                
                numPos = len(theLongs)
                totalLong = totalLong + numPos
                numNeg = len(theShorts)
                totalShort = totalShort + numNeg
                
                corLong = np.where(np.sign(testY[theLongs]) == 1)[0]
                longRet = np.sum(testY[theLongs])
                rollLongRight = rollLongRight + len(corLong)
                
                corShort = np.where(np.sign(testY[theShorts]) == -1)[0] 
                shortRet = np.sum(testY[theShorts])
                rollShortRight = rollShortRight + len(corShort)
                
                theRet = round(float(longRet) - float(shortRet),8)
                rollRet = round(float(rollRet) + float(theRet),8)
                thePerf.append(theRet)
                tempStr = pd.DataFrame({'ticker': [theTickers[i]],'theRet': [theRet],'rollret': [rollRet],'RollShortTrd': [totalShort],'RollLongTrd': [totalLong],
                'RollLongRight': [rollLongRight],'RollShortRight': [rollShortRight],'startTrainDate': [startTrainDate],'startTestDate': [startTestDate]})
                finalData = finalData.append(tempStr)
                print(theTickers[i] + " Ret: " + str(theRet) + " Roll Ret: " + str(rollRet) + " Short Cnt: " + str(numNeg) + " Long Cnt: " 
                + str(numPos) + " Strt Tr: " + startTrainDate + " Strt Test: " + startTestDate + " sRollCnt: " + str(totalShort) + " lRollCnt: " +  str(totalLong))
            except Exception, (e):
                print(theTickers[i])         
            pass
    except Exception, (e):
        print(theTickers[i])         
    pass      

finalData.to_csv(thePath + baseTicker + "_finalData.csv",index=False)
print("Sharpe: " + str(((np.mean(thePerf)/np.std(thePerf))*math.sqrt(252))))
temp = np.where(np.asarray(thePerf) < 0)
print("Sortino: " + str(np.mean(thePerf)/np.std(np.asarray(thePerf)[temp])*math.sqrt(252)))