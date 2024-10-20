import pandas as pd 
from datetime import date
import yfinance as yf
import streamlit as st 
import numpy as np
from yahoofinancials import YahooFinancials
import pandas_ta as ta
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 
from math import sqrt
from sklearn.ensemble import RandomForestClassifier  


classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")  

def process(token):
    df = yf.download(token,
                 start='2000-01-01',
                 end=date.today(),
                 progress=False,)
    df['RSI(2)'] = ta.rsi(df['Close'],length=2)
    df['RSI(7)'] = ta.rsi(df['Close'],length=7)
    df['RSI(14)'] = ta.rsi(df['Close'],length=14)
    df['CCI(30)'] = ta.cci(close=df['Close'],length=30,high=df['High'],low=df['Low'])
    df['CCI(50)'] = ta.cci(close=df['Close'],length=50,high=df['High'],low=df['Low'])
    df['CCI(100)'] = ta.cci(close=df['Close'],length=100,high=df['High'],low=df['Low'])
    df['LABEL'] = np.where( df['Open'].shift(-2).gt(df['Open'].shift(-1)),"1","0")
    df=df.dropna()
    X = df[df.columns[6:-1]].values 
    y = df['LABEL'].values
    model=classifier.fit(X,y) 
    return model
    