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
df = df.sort_values(by="Assignee")
#df = df.sort_values(by='AppDate')



# Saves the filtered inventor dataframe as csv
#df.to_csv('inventor-patent-sort-by-assignee.csv')
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
        

print("Length Company Names")
print(len(companyNames))

print("Length df")
print(len(df))

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

print(len(companyTickers))
print(len(set(companyTickers)))

# Creates a new column in the df with the ticker values
df['Tickers'] = companyTickers
print("Length of tickers merged with df")
print(len(df['Tickers']))
print("Length of df")
print(len(df))
print(len(df['Tickers']))

# Drops rows from df with no ticker data
df = df.dropna()
print("Length of df after dropping na")
print(len(df))
df.to_csv('inventor-patent-tickers.csv')

print("This is our inventor dataset")
print(df)

df.to_csv('inventor-patent-tickers-full.csv')

dates_list = df['AppDate'].to_numpy()

# Creates lists to store dates to be used to gather stock prices
month_before_app_date = []
app_date = []
month_after_app_date = []

ticker_list = df['Tickers'].to_numpy()
# GET DATES BEFORE AND AFTER
#ticker_list = closing_price_prior.to_numpy()
#ticker_list = closing_price_next.to_numpy()

# GET DATES FIRST, THEN DO THIS FOR 1mo INTERVAL WITH START DATE MONTH BEFORE AND END DATE A MONTH AFTER, THEN DO COMPARISON
#trying to get histories for the tickers
hist_list=[]
counter = 0

# Removes uplicates from the list
ticker_list_set = set(ticker_list)
ticker_list = list(ticker_list_set)
ticker_list.sort()
print("Length of ticker_list after set")
print(len(ticker_list))

app_year_sorted = list(df['AppYear'])
app_year_sorted.sort()
print("Earliest date")
print(app_year_sorted[0])

# This gets the earliest date to start getting stock data from
#earliest_date = str(app_year_sorted[0])[0:4] + "-01-01"
#print(earliest_date)

#TESTING TESTING TESTING REMOVE FOR FINAL VERSION!!!
#ticker_list = ticker_list[0:30]

print("Retrieiving stock data")
for i in ticker_list:
    if counter % 50 == 0:
        #print("Progress:", (counter/(len(ticker_list))) * 100, "%")
        print(counter)
    stock = yf.Ticker(i)
    # Sets period to get stock data from and retrieves it
    hist = stock.history(period="max", interval="1d")["Close"]
    hist_list.append(hist)
    counter = counter + 1
print("this is history of only the first ticker")
#an_history = hist_list[0]
an_history = hist_list[len(hist_list) - 1]
print(type(hist_list[0]))
print("History")
print(an_history)
print("History keys")
print(an_history.keys())
print("History index of first entry")
print(hist_list[0].index)

print("historical info is of type" + str(type(hist_list[0])))

print("Length of list that holds ticker history")
print(len(hist_list))
#print("Length of list that holds tickers")
#print(len(ticker_list))


closing_price_prior=[]
closing_price_current=[]
closing_price_next=[]

prev_dates = []
of_dates = []
next_dates = []

dates_list = df['AppDate'].to_numpy()

'''
for i in range(len(hist_list)):
    hist_list[i] = str(hist_list[i])
    hist_list[i].sort()
'''
"""
for i in range(len(dates_list)):
    if i % 1000 == 0:
        print(i)
    ticker = list(df["Tickers"])[i]
    ticker_index = BinarySearch(ticker_list, ticker)
    if ticker_index != -1:
        history = hist_list[ticker_index]
        curr_date = dates_list[i]
        #convert date to the correct format so we can match to find closing price, the problem is that
        #date formats arent consistent within dates_list
        if "/" in curr_date:
            try:
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
                of_dates.append(curr_date)
                prev_date = curr_date.replace(curr_month, prev_month)
                prev_date = prev_date.replace(curr_year, prev_year)
                prev_dates.append(prev_date)
                next_date = curr_date.replace(curr_month, next_month)
                next_date = next_date.replace(curr_year, next_year)
                next_dates.append(next_date)

                #if ((curr_date in history.index) and (next_date in history.index) and (prev_date in history.index)):
                #bin_value_before = BinarySearch(str(history.index), str(prev_date))
                #bin_value_of = BinarySearch(str(history.index), str(curr_date))
                #bin_value_after = BinarySearch(str(history.index), str(next_date))
                bin_value_before = BinarySearch(str(history[ticker_index]), str(prev_date))
                bin_value_of = BinarySearch(str(history[ticker_index]), str(curr_date))
                bin_value_after = BinarySearch(str(history[ticker_index]), str(next_date))
                if bin_value_before != -1 and bin_value_of != -1 and bin_value_after != -1:
                    closing_price_current.append(history.at[bin_value_of, 'Close'])
                    closing_price_next.append(history.at[bin_value_after, 'Close'])
                    closing_price_prior.append(history.at[bin_value_before, 'Close'])
                else:
                   closing_price_current.append(np.nan)
                    closing_price_next.append(np.nan)
                    closing_price_prior.append(np.nan)
            # If an error is thrown, we add nan to the list
            except:
                closing_price_current.append(np.nan)
                closing_price_next.append(np.nan)
                closing_price_prior.append(np.nan)
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
                of_dates.append(curr_date)
                prev_date = curr_date.replace(curr_month, prev_month)
                prev_date = prev_date.replace(curr_year, prev_year)
                prev_dates.append(prev_date)
                next_date = curr_date.replace(curr_month, next_month)
                next_date = next_date.replace(curr_year, next_year)
                next_dates.append(next_date)
                #bin_value_before = BinarySearch(str(history.index), str(prev_date))
                #bin_value_of = BinarySearch(str(history.index), str(curr_date))
                #bin_value_after = BinarySearch(str(history.index), str(next_date))
                bin_value_before = BinarySearch(str(history[ticker_index]), str(prev_date))
                bin_value_of = BinarySearch(str(history[ticker_index]), str(curr_date))
                bin_value_after = BinarySearch(str(history[ticker_index]), str(next_date))
                if bin_value_before != -1 and bin_value_of != -1 and bin_value_after != -1:
                    closing_price_current.append(history.at[bin_value_of, 'Close'])
                    closing_price_next.append(history.at[bin_value_after, 'Close'])
                    closing_price_prior.append(history.at[bin_value_before, 'Close'])
                else:
                    closing_price_current.append(np.nan)
                    closing_price_next.append(np.nan)
                    closing_price_prior.append(np.nan)
            except:
                closing_price_current.append(np.nan)
                closing_price_next.append(np.nan)
                closing_price_prior.append(np.nan)
    else:
        closing_price_current.append(np.nan)
        closing_price_next.append(np.nan)
        closing_price_prior.append(np.nan)
    #print(closing_price_next)
"""

for i in range(len(dates_list)):
    if i % 100000 == 0:
        print(i)
    curr_date = dates_list[i]
        #convert date to the correct format so we can match to find closing price, the problem is that
        #date formats arent consistent within dates_list
    if "/" in curr_date:
        try:
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


ticker_col = list(df["Tickers"])

counter = 0
hist_list_str = []

for i in range(len(hist_list)):
    hist_list_str.append([])
    for j in range(len(hist_list[i].index)):
        hist_list_str[i].append(str(hist_list[i].index[j])[0:10])

marker = 0

for i in range(len(dates_list)):
    if i % 100000 == 0:
        print(i)
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
                marker = i
                counter = counter+1
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

print(closing_price_current[marker])
print(closing_price_next[marker])
print(closing_price_prior[marker])
print(counter)
print(len(closing_price_next))
df['Month Before Application Date'] = prev_dates
df['Application Date'] = of_dates
df['Month After Application Date'] = prev_dates
df['Closing Price Last Month'] = closing_price_prior
df['Closing Price'] = closing_price_current
df['Closing Price Next Month'] = closing_price_next
df = df.dropna()
print("this is df")
print(df)

df.to_csv('inventor-patent-tickers-prices-full.csv')


