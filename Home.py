import pandas as pd 
from datetime import date, time
import yfinance as yf
import streamlit as st
import train
from yahoofinancials import YahooFinancials 
import altair as alt
import numpy as np
import pandas_ta as ta
from dateutil.relativedelta import relativedelta


import warnings 
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Market Minds",
    page_icon="💲",
    layout="wide",
)

def v_spacer(height, sb=False) -> None:
    for _ in range(height):
        if sb:
            st.sidebar.write('\n')
        else:
            st.write('\n')

st.title("💲Market Minds!")
st.sidebar.write("**Your very own smart trader!**")
st.sidebar.write("Analyze the various indicators of any stock of your choice, and use our nifty Machine Learning MOdel that is trained on data starting from the year 2000, up until now! The model keeps learning and growing with each day, meaning it is equipped to deal with the ever changing dynamic paradigm that is the stock market, giving you informed decisions based on over 8000 datapoints and counting!")
df_for_dict=pd.read_csv('name_symbol.csv')[['Symbol','Name']]

st.write("**Know what you see!**")
st.write("While we do provide the tools to empower you, your own knowledge over said tools is what gets you a step above the rest.")
st.write("We do not use the rudimentary parameters like opening and closing prices or the daily highs and lows, but rather take into consideration the RSI(Relative Strength Index) and CCI(Commodity Channel Index) of each stock.")

st.title("**See what you read!**")
st.write("Use own cutomisable panel to visualize data on various parameters, the details of all parameters are given below.")
v_spacer(10,sb=True)
symbols=df_for_dict['Symbol'].to_list()
names=df_for_dict['Name'].to_list()
dictionary = dict(zip(names, symbols))

stock=st.sidebar.selectbox('Select the stock.', options=names)

stock_data = yf.download(dictionary[stock], start='2000-01-01', end=date.today(), progress=False)
df = stock_data.reset_index()
# df['Color'] = np.where(df['Close'] > df['Open'], 1, 0)
model=train.process(dictionary[stock])
# alt.Chart(df).mark_boxplot().encode(
#     x='Date:T',
#     y=alt.Y('Low:Q', title='Stock Value'),
#     y2='High:Q'
# ).properties(
#     title='Stock Value Distribution',
#     width=600
# )
df['RSI(2)'] = ta.rsi(df['Close'],length=2)
df['RSI(7)'] = ta.rsi(df['Close'],length=7)
df['RSI(14)'] = ta.rsi(df['Close'],length=14)
df['CCI(30)'] = ta.cci(close=df['Close'],length=30,high=df['High'],low=df['Low'])
df['CCI(50)'] = ta.cci(close=df['Close'],length=50,high=df['High'],low=df['Low'])
df['CCI(100)'] = ta.cci(close=df['Close'],length=100,high=df['High'],low=df['Low'])
df=df.dropna()
selected_columns = ['RSI(2)', 'RSI(7)', 'RSI(14)', 'CCI(30)', 'CCI(50)', 'CCI(100)']

X = [df[selected_columns].to_numpy()[-1]]
prediction=model.predict(X)
df_filtered = df[pd.to_datetime(df['Date']).dt.date >= date.today() - relativedelta(months=+4)]

choice=st.radio(label="Pick a method for graph visualisation.",options=['RSI(2)','RSI(7)','RSI(14)','CCI(30)','CCI(50)','CCI(100)'],horizontal=True)
choice_style=st.radio(label="Choose style of visualization.",options=['Bar Chart','Line Chart'],horizontal=True)
if choice_style=='Bar Chart':
    params_chart = alt.Chart(df_filtered).mark_bar().encode(
        x=alt.X('Date', title="Date"),  
        y=alt.Y(choice, title="RSI/CCI")
    )
else:
    params_chart = alt.Chart(df_filtered).mark_line().encode(
            x=alt.X('Date', title="Date"),  
            y=alt.Y(choice, title="RSI/CCI")
        )
st.altair_chart(params_chart, use_container_width=True)
v_spacer(1)
ans=None
if prediction==1:
    st.success("Taking into consideration data from the 1st of January of 2000, here's what we predict for the next period of this stock - **UP**")
else:
    st.error("Taking into consideration data from the 1st of January of 2000, here's what we predict for the next period of this stock - **DOWN**")

v_spacer(2)
st.title("**Know your terms!**")
st.subheader("**RSI - Relative Strength Index**")
st.write("The standard RSI is calculated based on the average gains and losses over a specified period, typically 14 periods. RSI(2) shortens this period to 2, making it a more sensitive indicator to short-term price changes. A lower period, such as RSI(2), can generate more frequent and potentially earlier signals, but it may also be more prone to false signals due to increased volatility.")
st.write("RSI may be calculated as:")
st.latex(r'''
         100 - 100/(1+RS)
         ''')
st.latex (r'''
          RS = Average Gain/Average Loss
          ''')
v_spacer(1)
st.subheader("**CCI - Commodity Channel Index**")
st.write("CCI stands for Commodity Channel Index, and it is a popular technical analysis indicator used in the stock market and other financial markets. Developed by Donald Lambert, CCI is designed to identify cyclical trends in a security's price and to detect overbought or oversold conditions.")
st.write("The Commodity Channel Index is calculated using the following formula:")

st.latex(r'''
    (TP - SMA)/(0.015*MD)
    ''')

st.write("**TP**  - is the typical price, calculated as (High+Low+Close)/3.")
st.write("**SMA** - is the simple moving average of the typical price over a specified period.")
st.write("**MD** - is the mean deviation, a measure of the average absolute difference between the typical price and the SMA.")