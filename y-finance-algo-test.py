# Authors: Group 7 - McKenzie Hawkins, Alexander Mazon, Haolin Hu

# Imports
import pandas as pd
import datetime
import numpy as np
from bisect import bisect_left
import yfinance as yf

# Methods
# Obtained from https://www.geeksforgeeks.org/binary-search-bisect-in-python/
def BinarySearch(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else:
        return -1

######################################
# Data Manipulation
######################################

print("Reading in dataset, dropping irrelvant columns, and dropping rows with empty values.")

# Reads in the dataset to a dataframe
df = pd.read_csv('invpat.csv', dtype={'Zipcode': str, 'AsgNum': str, 'Invnum_N_UC': str})

# Removes unecessary fields
df = df.drop(['Street', 'Lat', 'Lng', 'InvSeq', 'AsgNum', 'Class', 'Invnum', 'Invnum_N', 'Invnum_N_UC', 'Density', 'Precision', 'Recall'], axis = 1)
df = df.dropna()

print("Sorting datast by company to make comparisons faster")
df = df.sort_values(by='Assignee')

# Saves the filtered inventor dataframe as csv
df.to_csv('inventor-patent-sort-by-assignee.csv')
print(df)

######################################
# Stock Data Retrieval
######################################

# Reads in a seperate dataset with tickers and company names from Quandl
stocksDF = pd.read_csv('invpat/ticker_list.csv')

print("Sorting datast by company to make comparisons faster")
stocksDF = stocksDF.sort_values(by='Name')

# Prints the dataframe containing various stock names and tickers
print(stocksDF)

# Set up array for ticker values
companyTickers = []

#assigns arrays to proper fields in the dataset
companyNames = df["Assignee"].tolist()
stockNames = stocksDF["Name"].tolist()
tickers = stocksDF["Ticker"].tolist()

for i in range(len(companyNames)):
    if ' ' in companyNames[i]:
        # Checks if inc or corp are in the second word as they aren't always listed the same in both datasets
        if "inc".upper() in companyNames[i].split()[1].upper() or "corp".upper() in companyNames[i].split()[1].upper():
            companyNames[i] = companyNames[i].split()[0].upper()
        else:
            companyNames[i] = companyNames[i].split()[0].upper() + " " + companyNames[i].split()[1].upper()
    else:
        companyNames[i] = companyNames[i].upper()

for i in range(len(stockNames)):
    if ' ' in stockNames[i]:
        # Checks if inc or corp are in the second word as they aren't always listed the same in both datasets
        if "inc".upper() in stockNames[i].split()[1].upper() or "corp".upper() in stockNames[i].split()[1].upper():
            stockNames[i] = stockNames[i].split()[0].upper()
        else:
            stockNames[i] = stockNames[i].split()[0].upper() + " " + stockNames[i].split()[1].upper()
    else:
        stockNames[i] = stockNames[i].upper()

print("Length Company Names")
print(len(companyNames))

print("Length df")
print(len(df))

# Convert to set to make lookup more efficient
#stockNames = set(stockNames)

# Gets the tickers of the companies from the new stocksDF
for i in range(len(df)):
    # Prints out progress
    if i % 100000 == 0:
        print("Rows checked out of", len(df))
        print(i)
        # Checks if first word of assignee is equal to first word of a stock name
    bin_value = BinarySearch(stockNames, companyNames[i])
    if bin_value != -1:
        companyTickers.append(tickers[bin_value])
    # Else adds a nan
    else:
        companyTickers.append(np.nan)

"""
    if (companyNames[i] in stockNames):
        for index, name in enumerate(stockNames):
            # Then checks to see if the strings are equal
            if (companyNames[i] == name):
                companyTickers.append(tickers[index])
                break
"""

print(len(companyTickers))

# Creates a new column in the df with the ticker values
df['Tickers'] = companyTickers

# Drops rows from df with no ticker data
df = df.dropna()
df.to_csv('inventor-patent-tickers.csv')

print("This is our invetor dataset")
print(df)

df.to_csv('inventor-patent-tickers-full.csv')

dates_list = df['AppDate'].to_numpy()

# Creates lists to store dates to be used to gather stock prices
month_before_app_date = []
app_date = []
month_after_app_date = []

ticker_list = df['Tickers'].to_numpy()
# GET DATES BEFORE AND AFTER
ticker_list = df['Tickers'].to_numpy()
ticker_list = df['Tickers'].to_numpy()

# GET DATES FIRST, THEN DO THIS FOR 1mo INTERVAL WITH START DATE MONTH BEFORE AND END DATE A MONTH AFTER, THEN DO COMPARISON
#trying to get histories for the tickers
hist_list=[]
counter = 0
for i in ticker_list:
    if counter % 10000 == 0:
        print(counter)
    stock = yf.Ticker(i)
    hist = stock.history(period="1mo", )
    hist_list.append(hist)
    counter = counter + 1
print("this is history of only the first ticker")
#an_history = hist_list[0]
an_history = hist_list[len(hist_list) - 1]
print(an_history)
print(an_history.keys())

print("historical info is of type" + str(type(hist_list[0])))

print("Length of list that holds ticker history")
print(len(hist_list))
#print("Length of list that holds tickers")
#print(len(ticker_list))

print("this is dataframe")
print(df)
closing_price_prior=[]
closing_price_current=[]
closing_price_next=[]

dates_list = df['AppDate'].to_numpy()

for i in range(len(dates_list)):
    if i % 10000 == 0:
        print(i)
    history = hist_list[i]
    curr_date = dates_list[i]
    #convert date to the correct format so we can match to find closing price, the problem is that
    #date formats arent consistent within dates_list
    if "/" in curr_date:
        d = datetime.datetime.strptime(curr_date, "%m/%d/%Y")
        curr_date = datetime.date.strftime(d, '%Y-%m-%d')
        curr_year = int(curr_date[:4])
        curr_month = int(curr_date[5:7])
        #processing to get dates one month before and after patent
        if curr_month < 12:
            next_month = str(curr_month + 1)
            next_year = str(curr_year)
        else:
            next_month = str(1)
            next_year = str(curr_year + 1)
        if curr_month > 1:
            prev_month = str(curr_month - 1)
            prev_year = str(curr_year)
        else:
            prev_month = str(12)
            prev_year = str(curr_year - 1)
        curr_month = str(curr_month)
        curr_year = str(curr_year)
        prev_date = curr_date.replace(curr_month, prev_month)
        prev_date = prev_date.replace(curr_year, prev_year)
        next_date = curr_date.replace(curr_month, next_month)
        next_date = next_date.replace(curr_year, next_year)
        #if ((curr_date in history.index) and (next_date in history.index) and (prev_date in history.index)):
        bin_value_before = BinarySearch(history.index, prev_date)
        bin_value_of = BinarySearch(history.index, curr_date)
        bin_value_after = BinarySearch(history.index, next_date)
        if bin_value_before != -1 or bin_value_of != -1 or bin_value_after != -1:
            closing_price_current.append(history.at[bin_value_of, 'Close'])
            closing_price_next.append(history.at[bin_value_after, 'Close'])
            closing_price_prior.append(history.at[bin_value_before, 'Close'])
        else:
            closing_price_current.append(np.nan)
            closing_price_next.append(np.nan)
            closing_price_prior.append(np.nan)
        if (i == 235):
            print("i = 238 for LYTS stock test")
            print(curr_date)
            print(history.at["01/17/1994", 'Close'])
            print(history.at["02/17/1994", 'Close'])
            print(history.at["03/17/1994", 'Close'])
            print(history.at[prev_date, 'Close'])
            print(history.at[curr_date, 'Close'])
            print(history.at[next_date, 'Close'])
    else:
        curr_date_year = curr_date[0:4]
        curr_date_month = curr_date[4:6]
        curr_date_day = curr_date[6:8]
        curr_date = curr_date_month + "/" + curr_date_day + "/" + curr_date_year
        print(curr_date)
        d = datetime.datetime.strptime(curr_date, "%m/%d/%Y")
        curr_date = datetime.date.strftime(d, '%Y-%m-%d')
        curr_year = int(curr_date[:4])
        curr_month = int(curr_date[5:7])
        #processing to get dates one month before and after patent
        if curr_month < 12:
            next_month = str(curr_month + 1)
            next_year = str(curr_year)
        else:
            next_month = str(1)
            next_year = str(curr_year + 1)
        if curr_month > 1:
            prev_month = str(curr_month - 1)
            prev_year = str(curr_year)
        else:
            prev_month = str(12)
            prev_year = str(curr_year - 1)
        curr_month = str(curr_month)
        curr_year = str(curr_year)
        prev_date = curr_date.replace(curr_month, prev_month)
        prev_date = prev_date.replace(curr_year, prev_year)
        next_date = curr_date.replace(curr_month, next_month)
        next_date = next_date.replace(curr_year, next_year)
        bin_value_before = BinarySearch(history.index, prev_date)
        bin_value_of = BinarySearch(history.index, curr_date)
        bin_value_after = BinarySearch(history.index, next_date)
        if bin_value_before != -1 or bin_value_of != -1 or bin_value_after != -1:
            closing_price_current.append(history.at[bin_value_of, 'Close'])
            closing_price_next.append(history.at[bin_value_after, 'Close'])
            closing_price_prior.append(history.at[bin_value_before, 'Close'])
        else:
            closing_price_current.append(np.nan)
            closing_price_next.append(np.nan)
            closing_price_prior.append(np.nan)
    #print(closing_price_next)

print(closing_price_next)
#padding missing values with zeros
#while len(closing_price_current) < len(dates_list):
#    closing_price_current.append(0)
#while len(closing_price_next) < len(dates_list):
#    closing_price_next.append(0)
#while len(closing_price_prior) < len(dates_list):
#    closing_price_prior.append(0)
df['Closing Price'] = closing_price_current
df['Closing Price Last Month'] = closing_price_prior
df['Closing Price Next Month'] = closing_price_next
print("this is df")
print(df)

print(df)

