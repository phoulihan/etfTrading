# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:46:58 2016

@author: Patrick Houlihan
prepared for HRG to illustrate a simple alpha rich ETF strategy
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm
from decimal import Decimal
from pymongo import MongoClient
from sklearn import linear_model, datasets
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.svm import SVC
from sklearn import linear_model, datasets
import imp
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn import linear_model, datasets

thePath = 'C:/Users/xilin/Google Drive/iViewPrep/'

mongo = MongoClient('127.0.0.1', 27017)
mongoDb = mongo['priceData']
mongoColl = mongoDb['crspData']

theTickers = np.sort(np.array(mongoColl.distinct('ticker')))
numTickers = len(theTickers)

theMainEtf = "$UPRO"
startDate = "2014-01-01"
endDate = "2016-05-31"

etfData = pd.DataFrame(list(mongoColl.find({"$and":[{"date": {"$gte": startDate,"$lte": endDate}},{"ticker": theMainEtf}]},{"date": 1, "ticker": 1, "PRC": 1})))     
etfData['etfRet'] = np.log(etfData.PRC.astype(float)) - np.log(etfData.PRC.astype(float).shift(1))
etfData = etfData.drop(['_id','ticker'], 1)
etfData.rename(columns = {'PRC':'etfPrc'}, inplace = True)
etfDataLen = len(etfData)

theCorData = pd.DataFrame()
theStatData = pd.DataFrame()
finalData = pd.DataFrame()

theWindow = 30 #lagged correlation
rollRet = 0
testSize = .10
thePerf = list()
postThresh= .8

for i in range(0,numTickers):
#for i in range(0,100): 
    tempData = pd.DataFrame(list(mongoColl.find({"$and":[{"date": {"$gte": startDate,"$lte": endDate}},{"ticker": theTickers[i]}]},{"date": 1, "ticker": 1, "PRC": 1})))     
    if(len(tempData) > 0):
        tempData['equityRet'] = tempData.PRC.astype(float).pct_change()
        tempData = tempData.drop(['_id','ticker'], 1) 
        tempDataLen = len(tempData)
        tempData = tempData.merge(etfData,left_on='date', right_on='date', how='outer') 
        tempData['theDiff'] = tempData.etfPrc.astype(float) - tempData.PRC.astype(float).shift(1)
        tempData = tempData[['date','theDiff','etfRet','equityRet']].dropna()
        tempData= tempData.reindex()
        numSamples =len(tempData)
        if(tempDataLen == etfDataLen):
            theCorDataTemp = pd.DataFrame()
            theFeatures = np.zeros([1, 3], dtype="S40")
            for j in range(0,numSamples-theWindow):
                temp = np.zeros([1, 3], dtype=Decimal) 
                theWin = theWindow + j
                theCorTemp = pearsonr(tempData['etfRet'][j:theWin], tempData['equityRet'][j:theWin])
                sdDiff = tempData['theDiff'][j:theWin].std()
                kurtDiff = tempData['theDiff'][j:theWin].kurt()
                skewDiff = tempData['theDiff'][j:theWin].skew()
                tempStr = pd.DataFrame({'theDiff': [tempData['theDiff'][j+1]],'etfRet': [tempData['etfRet'][j+1]],'equityRet': [tempData['equityRet'][j+1]],'correl': theCorTemp[0]})
                theCorDataTemp = theCorDataTemp.append(tempStr)
                temp[0][0] = tempData['etfRet'][j+1]
                temp[0][1] = tempData['equityRet'][j+1]
                temp[0][2] = theCorTemp[0]
                theFeatures = np.concatenate((theFeatures,temp))
            y = theCorDataTemp['equityRet'][theWindow+1:len(theCorDataTemp)] #next day assset return
            #print((y - np.mean(y))/np.std(y))
            X = theCorDataTemp[['equityRet','etfRet','correl','theDiff']][theWindow:len(theCorDataTemp)-1] #event day features
            X = (X - np.mean(X))/np.std(X) #feature scaling
            #print((X - np.mean(X))/np.std(X))
            try:
                y.index = X.index.values     
                model = sm.OLS(y, X)
                result = model.fit()
                
                if(result.f_pvalue <= .05):
                    overallCor = pearsonr(tempData['etfRet'], tempData['equityRet'])
                    theStatData = theStatData.append({'ticker': theTickers[i],'p-value': result.f_pvalue,'correl': overallCor[0],'statSig': overallCor[1]}, ignore_index=True)
                    overallCor = pearsonr(tempData['etfRet'], tempData['equityRet'])
                    #model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=300,learning_rate=1,algorithm="SAMME",random_state=None)
                    model = RandomForestClassifier(n_estimators=1000,random_state=42)#RandomForestClassifier(n_estimators=10,max_depth=None,min_samples_split=1,random_state=531)
                    #model = SVC(kernel='rbf', class_weight=None, probability=True, random_state=42)
                    #model = linear_model.LogisticRegression(C=1e5)
                    trainY, testY = train_test_split(np.asarray(y,dtype="|S10"), test_size = testSize)
                    trainX, testX = train_test_split(np.asarray(X,dtype="|S10"), test_size = testSize)
                    model.fit(trainX,np.sign(trainY.astype(float)))

                    postProbs = model.predict_proba(testX) 

                    theClasses = model.classes_ #posterior column class alignment -1,0,1?
                    #print(theClasses)
                      
                    theShorts = np.where(postProbs[:,0] >= postThresh) #SHORT POSITIONS

                    try:                    
                        theLongs = np.where(postProbs[:,2] >= postThresh) #LONG POSITIONS
                    except Exception, (e): 
                        theLongs = np.where(postProbs[:,1] >= postThresh) #LONG POSITIONS
                        print(e)  
                    pass
   
                    shortTrd = np.size(theLongs)
                    longTrd = np.size(theShorts)
                    
                    corLong = np.where(np.sign(testY[theLongs].astype(float)) == 1)
                    incLong = np.where(np.sign(testY[theLongs].astype(float)) == -1) 
                    longRet = np.sum(testY[corLong].astype(float)) - np.sum(testY[incLong].astype(float))
                    
                    corShort = np.where(np.sign(testY[theShorts].astype(float)) == -1)
                    incShort = np.where(np.sign(testY[theShorts].astype(float)) == 1)
                    shortRet = -np.sum(testY[corShort].astype(float)) + np.sum(testY[incShort].astype(float))
                        
                    theRet = longRet + shortRet
                    rollRet = rollRet + theRet
                    print(theTickers[i] + ": eqRet: " +str(theRet) + " rollRet: " + str(rollRet) + " numLong: " + str(longTrd) + " numShort: " + str(shortTrd))
                    thePerf.append(round(theRet,6))
                        
                    tempStr = pd.DataFrame({'ticker': [theTickers[i]],'p-value': [result.f_pvalue],'correl': [theCorTemp[0]], 'corStatSig': [overallCor[1]],'rSqr': [result.rsquared],'theRet': [theRet],'rollret': [rollRet],'shortTrd': [shortTrd],'longTrd': [longTrd]})
                    finalData = finalData.append(tempStr)
            except Exception, (e):
                print(e)         
            pass  

print("Sharpe: " + str(np.mean(thePerf)/np.std(thePerf)))
temp = np.where(np.asarray(thePerf) < 0)
print("Sortino: " + str(np.mean(thePerf)/np.std(np.asarray(thePerf)[temp])))

finalData.to_csv(thePath + "finalData.csv",index=False)

#http://www.analyticsvidhya.com/blog/2015/09/naive-bayes-explained/
#http://statsmodels.sourceforge.net/devel/generated/statsmodels.regression.linear_model.RegressionResults.html
#http://stackoverflow.com/questions/37508158/how-to-extract-a-particular-value-from-the-ols-summary-in-pandas