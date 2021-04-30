# Authors: Group 7 - McKenzie Hawkins, Alexander Mazon, Haolin Hu

# Imports
import pandas as pd
import datetime
import numpy as np
import yfinance as yf
#import pandas_datareader.data as web

# Reads in the dataset to a dataframe
df = pd.read_csv('tickers_and_dates_before_of_and_after.csv')

# Gets the list of tickers from the dataset
tickers = df["Tickers"].to_numpy()

# Gets the dates from the dataset to get stock prices from
month_before_app_date = df["Month Before Application Date"].to_numpy()
app_date = df["Application Date"].to_numpy()
month_after_app_date = df["Month After Application Date"].to_numpy()

yf_tickers = []
price_month_before = []
price_day_of = []
price_month_after = []

# YFINANCE SECTION (ISN'T WORKING)
# Gets the yfinance data for each ticker in the list
for i in range(len(tickers)):
    if i % 10000 == 0:
        print("Progress to", len(tickers))
        print(i)
    yf_tickers.append(yf.Ticker(tickers[i]))

# Gets the stock price for the date in each list
#for i in range(len(month_before_app_date)):
for i in range(100):
    if i % 10000 == 0:
        print(i)
    try:
        price_month_before.append(yf_tickers[i].history(period='1d', start=str(month_before_app_date[i]), end=str(month_before_app_date[i])))
    except:
        price_month_before.append(np.nan)

print(price_month_before)
'''
price_month_before_test = []
price_day_of_test = []
price_month_after_test = []

# PANDAS DATAREADER SECTION
for i in range(len(tickers)):
    if i % 10000 == 0:
        print("Progress to ", len(tickers))
        print(i)
    data = web.DataReader(name=tickers[i], data_source='yahoo', start=month_before_app_date[i], end=month_before_app_date[i])['Close']
print(data)
'''