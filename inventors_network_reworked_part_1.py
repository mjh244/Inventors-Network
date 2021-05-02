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
df = pd.read_csv('datasets/invpat.csv', dtype={'Zipcode': str, 'AsgNum': str, 'Invnum_N_UC': str})

# Removes unecessary fields
df = df.drop(['Street', 'Lat', 'Lng', 'InvSeq', 'AsgNum', 'Class', 'Invnum', 'Invnum_N', 'Invnum_N_UC', 'Density', 'Precision', 'Recall'], axis = 1)
df = df.dropna()

print("Sorting datast by company to make comparisons faster")
df = df.sort_values(by="Assignee")

# Saves the filtered inventor dataframe as csv
df.to_csv('datasets/inventor-patent-sort-by-assignee.csv')
print(df)

######################################
# Stock Data Retrieval
######################################

# Reads in a seperate dataset with tickers and company names from Quandl
stocksDF = pd.read_csv('datasets/ticker_list.csv')

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

# Formats assignee and company names across the two datasets to make matching easier
for i in range(len(companyNames)):
    companyNames[i] = companyNames[i].upper()
    if ' ' in companyNames[i]:
        # Checks if inc or corp are in the second word as they aren't always listed the same in both datasets
        if "INC" in companyNames[i].split()[1] or "CORP" in companyNames[i].split()[1]:
            companyNames[i] = companyNames[i].split()[0]
        else:
            companyNames[i] = companyNames[i].split()[0] + " " + companyNames[i].split()[1]

for i in range(len(stockNames)):
    stockNames[i] = stockNames[i].upper()
    if ' ' in stockNames[i]:
        # Checks if inc or corp are in the second word as they aren't always listed the same in both datasets
        if "INC" in stockNames[i].split()[1] or "CORP" in stockNames[i].split()[1]:
            stockNames[i] = stockNames[i].split()[0]
        else:
            stockNames[i] = stockNames[i].split()[0] + " " + stockNames[i].split()[1]


print("Matching assignee names to public company names to retrieve tickers")
# Gets the tickers of the companies from the new stocksDF
for i in range(len(df)):
    # Checks if an assignee matches to a public company name
    bin_value = BinarySearch(stockNames, companyNames[i])
    # Store the ticker if there is a name match
    if bin_value != -1:
        companyTickers.append(tickers[bin_value])
    # Else adds a nan
    else:
        companyTickers.append(np.nan)

print("Number of tickers", len(companyTickers))
print("Number of unique tickers", len(set(companyTickers)))

# Creates a new column in the df with the ticker values
df['Tickers'] = companyTickers

# Drops rows from df with no ticker data
df = df.dropna()
df.to_csv('datasets/inventor-patent-tickers.csv')

print("This is our inventor dataset")
print(df)

df.to_csv('datasets/inventor-patent-tickers-full.csv')

dates_list = df['AppDate'].to_numpy()

# Creates lists to store dates to be used to gather stock prices
month_before_app_date = []
app_date = []
month_after_app_date = []

ticker_list = df['Tickers'].to_numpy()

# Removes uplicates from the list
ticker_list_set = set(ticker_list)
ticker_list = list(ticker_list_set)
ticker_list.sort()
print("Length of ticker_list after set")
print(len(ticker_list))

hist_list=[]
counter = 0

print("Retrieiving stock data")
for i in ticker_list:
    # Prints progress on donwloading stock prices
    if counter % 50 == 0:
        print("Stock price download progress:", ((counter/len(ticker_list)) * 100), "%")
        print(counter)
    stock = yf.Ticker(i)
    # Sets period to get stock data from and retrieves it
    hist = stock.history(period="max", interval="1d")["Close"]
    hist_list.append(hist)
    counter = counter + 1
print("Stock price download progress: 100 %")

# Initializes lists to store dates
prev_dates = []
of_dates = []
next_dates = []

dates_list = df['AppDate'].to_numpy()

# Attempts to standardize date format
for i in range(len(dates_list)):
    if i % 100000 == 0:
        print("Date formatting progress:", ((i/len(dates_list)) * 100), "%")
    curr_date = dates_list[i]
        # Converts original date to the correct format so we can match to find closing price
        # Many date formats aren't consistent within original dates_list
    if "/" in curr_date:
        try:
            d = datetime.datetime.strptime(curr_date, "%m/%d/%Y")
            curr_date = datetime.date.strftime(d, '%Y-%m-%d')
            curr_year = int(curr_date[:4])
            curr_month = int(curr_date[5:7])
            # Processing to get dates one month before and after patent
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
            of_dates.append(curr_date)
            prev_date = curr_date.replace(curr_month, prev_month)
            prev_date = prev_date.replace(curr_year, prev_year)
            prev_dates.append(prev_date)
            next_date = curr_date.replace(curr_month, next_month)
            next_date = next_date.replace(curr_year, next_year)
            next_dates.append(next_date)
        except:
            of_dates.append(np.nan)
            prev_dates.append(np.nan)
            next_dates.append(np.nan)
    else:
        try:
            curr_date_year = curr_date[0:4]
            curr_date_month = curr_date[4:6]
            curr_date_day = curr_date[6:8]
            curr_date = curr_date_month + "/" + curr_date_day + "/" + curr_date_year
            d = datetime.datetime.strptime(curr_date, "%m/%d/%Y")
            curr_date = datetime.date.strftime(d, '%Y-%m-%d')
            curr_year = int(curr_date[:4])
            curr_month = int(curr_date[5:7])
            # Processing to get dates one month before and after patent
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
            of_dates.append(curr_date)
            prev_date = curr_date.replace(curr_month, prev_month)
            prev_date = prev_date.replace(curr_year, prev_year)
            prev_dates.append(prev_date)
            next_date = curr_date.replace(curr_month, next_month)
            next_date = next_date.replace(curr_year, next_year)
            next_dates.append(next_date)
        except:
            of_dates.append(np.nan)
            prev_dates.append(np.nan)
            next_dates.append(np.nan)
print("Date formatting progress: 100 %")

ticker_col = list(df["Tickers"])

hist_list_str = []

# Converts the datetimeindex objects to strings
for i in range(len(hist_list)):
    hist_list_str.append([])
    for j in range(len(hist_list[i].index)):
        hist_list_str[i].append(str(hist_list[i].index[j])[0:10])

# Initializes lists to store stock prices at specific dates
closing_price_prior=[]
closing_price_current=[]
closing_price_next=[]

# Finds the stock price based on the retrieved histories.
for i in range(len(dates_list)):
    if i % 100000 == 0:
        print("Price gathering progress:", ((i/len(dates_list)) * 100), "%")
    try:
        ticker_index = BinarySearch(ticker_list, ticker_col[i])
        if ticker_index != -1:
            history = hist_list[ticker_index]
            bin_value_before = BinarySearch(hist_list_str[ticker_index], prev_dates[i])
            bin_value_of = BinarySearch(hist_list_str[ticker_index], of_dates[i])
            bin_value_after = BinarySearch(hist_list_str[ticker_index], next_dates[i])
            if bin_value_before != -1 and bin_value_of != -1 and bin_value_after != -1:
                closing_price_current.append(history[bin_value_of])
                closing_price_next.append(history[bin_value_after])
                closing_price_prior.append(history[bin_value_before])
            else:
                closing_price_current.append(np.nan)
                closing_price_next.append(np.nan)
                closing_price_prior.append(np.nan)
        else:
            closing_price_current.append(np.nan)
            closing_price_next.append(np.nan)
            closing_price_prior.append(np.nan)
    except:
        closing_price_current.append(np.nan)
        closing_price_next.append(np.nan)
        closing_price_prior.append(np.nan)
print("Price gathering progress: 100 %")

# Merges the lists with the original dataframe
df['Month Before Application Date'] = prev_dates
df['Application Date'] = of_dates
df['Month After Application Date'] = prev_dates
df['Closing Price Last Month'] = closing_price_prior
df['Closing Price'] = closing_price_current
df['Closing Price Next Month'] = closing_price_next
df = df.dropna()
print("Current Inventor Dataset")
print(df)
df.to_csv('datasets/inventor-patent-tickers-prices-full.csv')


