# Authors: Group 7 - McKenzie Hawkins, Alexander Mazon, Haolin Hu

# Imports
import pandas as pd
import numpy as np
from bisect import bisect_left
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import yfinance as yf

# Methods
# Obtained from https://www.geeksforgeeks.org/binary-search-bisect-in-python/
# Used to perform quicker searching
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

print("Sorting inventor datast by assignees to make comparisons faster")
df = df.sort_values(by="Assignee")

print(df)

######################################
# Stock Data Retrieval
######################################

# Reads in a seperate dataset with tickers and company names from Quandl
stocksDF = pd.read_csv('datasets/ticker_list.csv')

print("Sorting ticker datast by names to make comparisons faster")
stocksDF = stocksDF.sort_values(by='Name')

# Prints the dataframe containing various stock names and tickers
print(stocksDF)

# Set up array for ticker values
companyTickers = []

#assigns arrays to proper fields in the dataset
companyNames = df["Assignee"].tolist()
stockNames = stocksDF["Name"].tolist()

# Formats assignee names to make matching easier
for i in range(len(companyNames)):
    companyNames[i] = companyNames[i].upper()
    if ' ' in companyNames[i]:
        # Checks if inc or corp are in the second word as they aren't always listed the same in both datasets
        #if "INC" in companyNames[i].split()[1] or "CORP" in companyNames[i].split()[1] or "LTD" in companyNames[i].split()[1]:
        if "INC" in companyNames[i].split()[1] or "CORP" in companyNames[i].split()[1]:
            companyNames[i] = companyNames[i].split()[0]
        else:
            companyNames[i] = companyNames[i].split()[0] + " " + companyNames[i].split()[1]

# Formats stock comapny names to make matching easier
for i in range(len(stockNames)):
    stockNames[i] = stockNames[i].upper()
    if ' ' in stockNames[i]:
        # Checks if inc or corp are in the second word as they aren't always listed the same in both datasets
        #if "INC" in stockNames[i].split()[1] or "CORP" in stockNames[i].split()[1] or "LTD" in stockNames[i].split()[1]:
        if "INC" in stockNames[i].split()[1] or "CORP" in stockNames[i].split()[1]:
            stockNames[i] = stockNames[i].split()[0]
        else:
            stockNames[i] = stockNames[i].split()[0] + " " + stockNames[i].split()[1]


# This reassigns the Assignee column to the new filtered company Names
df["Assignee"] = companyNames
# We then resort the dataframe by those Assignee names to use binary search
df = df.sort_values(by="Assignee")
# Then we grab those names
companyNames = df["Assignee"].tolist()

# This reassigns the Name column to the new filtered stock Names
stocksDF["Name"] = stockNames
# We then resort the dataframe by those Name names to use binary search
stocksDF = stocksDF.sort_values(by="Name")
# Then we grab those names
stockNames = stocksDF["Name"].tolist()

print(companyNames[0:10])
print(stockNames[0:10])

# Assigns the tickers list to the Ticker column in the Quotemedia ticker dataset
tickers = stocksDF["Ticker"].tolist()

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

print("This is our inventor dataset after name matching for stock tickers")
print(df)

dates_list = df['AppDate'].to_numpy()

ticker_list = df['Tickers'].to_numpy()

# Removes duplicates from the list and sorts it to be used for efficient searching
ticker_list_set = set(ticker_list)
ticker_list = list(ticker_list_set)
ticker_list.sort()
print("Number of unique tickers", len(ticker_list))

# Initializes list to store historical stock dates and prices for tickers
hist_list=[]
counter = 0

print("Retrieiving stock data")
for i in ticker_list:
    # Prints progress on donwloading stock prices
    if counter % 50 == 0:
        print("Stock price download progress:", ((counter/len(ticker_list)) * 100), "%")
    stock = yf.Ticker(i)
    # Sets period to get closing stock price from and retrieves it
    hist = stock.history(period="max", interval="1d")["Close"]
    hist_list.append(hist)
    counter = counter + 1
print("Stock price download progress: 100 % \n")

# Initializes lists to store dates
prev_dates_day_before = []
prev_dates = []
prev_dates_day_after = []
of_dates_day_before = []
of_dates = []
of_dates_day_after = []
next_dates_day_before = []
next_dates = []
next_dates_day_after = []

# Gets the application dates from the df and stores it
dates_list = df['AppDate'].to_numpy()

# Attempts to standardize date format
for i in range(len(dates_list)):
    if i % 100000 == 0:
        print("Date formatting progress:", ((i/len(dates_list)) * 100), "%")
    app_date = dates_list[i]
    
    # Converts the date to one with slashes to match earlier code
    if "/" not in app_date:
        curr_date_year = app_date[0:4]
        curr_date_month = app_date[4:6]
        curr_date_day = app_date[6:8]
        app_date = curr_date_month + "/" + curr_date_day + "/" + curr_date_year

    # Converts original date to the correct format so we can match to find closing price
    # Many date formats aren't consistent within original dates_list
    try:
        # Gets day of, before, and after patent application date
        date_of_app = datetime.strptime(app_date, "%m/%d/%Y")
        date_day_before_app = date_of_app + timedelta(-1)
        date_day_after_app = date_of_app + timedelta(1)
        
        # Gets month before application date, as well as day before and after that
        month_before_day_before_app_date = date_day_before_app - relativedelta(months=1)
        month_before_app_date = date_of_app - relativedelta(months=1)
        month_before_day_after_app_date = date_day_after_app - relativedelta(months=1)

        # Gets month after application date, as well as day before and after that
        month_after_day_before_app_date = date_day_before_app + relativedelta(months=1)
        month_after_app_date = date_of_app + relativedelta(months=1)
        month_after_day_after_app_date = date_day_after_app + relativedelta(months=1)

        # Converts the dates to a string with a standard format of year, then month, then day
        month_before_day_before_app_date = datetime.strftime(month_before_day_before_app_date, '%Y-%m-%d')
        month_before_app_date = datetime.strftime(month_before_app_date, '%Y-%m-%d')
        month_before_day_after_app_date = datetime.strftime(month_before_day_after_app_date, '%Y-%m-%d')
        date_day_before_app = datetime.strftime(date_day_before_app, '%Y-%m-%d')
        date_of_app = datetime.strftime(date_of_app, '%Y-%m-%d')
        date_day_after_app = datetime.strftime(date_day_after_app, '%Y-%m-%d')
        month_after_day_before_app_date = datetime.strftime(month_after_day_before_app_date, '%Y-%m-%d')
        month_after_app_date = datetime.strftime(month_after_app_date, '%Y-%m-%d')
        month_after_day_after_app_date = datetime.strftime(month_after_day_after_app_date, '%Y-%m-%d')

        # Appends the dates to their appropriate list to be used as a backup date to check when getting stock prices
        prev_dates_day_before.append(month_before_day_before_app_date)
        prev_dates.append(month_before_app_date)
        prev_dates_day_after.append(month_before_day_after_app_date)
        of_dates_day_before.append(date_day_before_app)
        of_dates.append(date_of_app)
        of_dates_day_after.append(date_day_after_app)
        next_dates_day_before.append(month_after_day_before_app_date)
        next_dates.append(month_after_app_date)
        next_dates_day_after.append(month_after_day_after_app_date)

    # If something goes wrong, just add nans to each list
    except:
        prev_dates_day_before.append(np.nan)
        prev_dates.append(np.nan)
        prev_dates_day_after.append(np.nan)
        of_dates_day_before.append(np.nan)
        of_dates.append(np.nan)
        of_dates_day_after.append(np.nan)
        next_dates_day_before.append(np.nan)
        next_dates.append(np.nan)
        next_dates_day_after.append(np.nan)

print("Date formatting progress: 100 % \n")

# Initializes lists to store stock dates as strings to be searched
hist_list_str = []

print("Converting dates to strings")
# Converts the datetimeindex objects to strings
for i in range(len(hist_list)):
    hist_list_str.append([])
    for j in range(len(hist_list[i].index)):
        hist_list_str[i].append(str(hist_list[i].index[j])[0:10])

# Gets tickers from df and stores it in a list
ticker_col = list(df["Tickers"])

# Initializes lists to store dates that stock prices were gathered at
date_prior=[]
date_current=[]
date_next=[]

# Initializes lists to store stock prices at specific dates
closing_price_prior=[]
closing_price_current=[]
closing_price_next=[]

# Finds the stock price based on the retrieved histories
for i in range(len(dates_list)):
    if i % 100000 == 0:
        print("Price gathering progress:", ((i/len(dates_list)) * 100), "%")
    try:
        ticker_index = BinarySearch(ticker_list, ticker_col[i])
        # If the ticker is in the ticker_list, see if the dates are in there as well
        if ticker_index != -1:
            # Gets history of current ticker
            history = hist_list[ticker_index]
            # Gets index of dates in lists to get prices
            bin_value_month_before_day_before = BinarySearch(hist_list_str[ticker_index], prev_dates_day_before[i])
            bin_value_month_before = BinarySearch(hist_list_str[ticker_index], prev_dates[i])
            bin_value_month_before_day_after = BinarySearch(hist_list_str[ticker_index], prev_dates_day_after[i])
            bin_value_of_day_before = BinarySearch(hist_list_str[ticker_index], of_dates_day_before[i])
            bin_value_of = BinarySearch(hist_list_str[ticker_index], of_dates[i])
            bin_value_of_day_after = BinarySearch(hist_list_str[ticker_index], of_dates_day_after[i])
            bin_value_month_after_day_before = BinarySearch(hist_list_str[ticker_index], next_dates_day_before[i])
            bin_value_month_after = BinarySearch(hist_list_str[ticker_index], next_dates[i])
            bin_value_month_after_day_after = BinarySearch(hist_list_str[ticker_index], next_dates_day_after[i])

            # If there is a stock price for a date a month before the app date, then store it
            if bin_value_month_before != -1:
                closing_price_prior.append(history[bin_value_month_before])
                date_prior.append(prev_dates[i])
            # Else if there is a stock price for a date a month and day before the app date, then store it
            elif bin_value_month_before_day_before != -1:
                closing_price_prior.append(history[bin_value_month_before_day_before])
                date_prior.append(prev_dates_day_before[i])
            # Else if there is a stock price for a date a month and day after the app date, then store it
            elif bin_value_month_before_day_after != -1:
                closing_price_prior.append(history[bin_value_month_before_day_after])
                date_prior.append(prev_dates_day_after[i])
            # Else we don't have data on any of the 3 dates so just add nans to the price and date list to merge with df
            else:
                closing_price_prior.append(np.nan)
                date_prior.append(np.nan)

            # If there is a stock price for the app date, then store it
            if bin_value_of != -1:
                closing_price_current.append(history[bin_value_of])
                date_current.append(of_dates[i])
            # Else if there is a stock price for a day before the app date, then store it
            elif bin_value_of_day_before != -1:
                closing_price_current.append(history[bin_value_of_day_before])
                date_current.append(of_dates_day_before[i])
            # Else if there is a stock price for a day after the app date, then store it
            elif bin_value_of_day_after != -1:
                closing_price_current.append(history[bin_value_of_day_after])
                date_current.append(of_dates_day_after[i])
            # Else we don't have data on any of the 3 dates so just add nans to the price and date list to merge with df
            else:
                closing_price_current.append(np.nan)
                date_current.append(np.nan)

            # If there is a stock price for a date a month after the app date, then store it
            if bin_value_month_after != -1:
                closing_price_next.append(history[bin_value_month_after])
                date_next.append(next_dates[i])
            # Else if there is a stock price for a date a month after and day before the app date, then store it
            elif bin_value_month_after_day_before != -1:
                closing_price_next.append(history[bin_value_month_after_day_before])
                date_next.append(next_dates_day_before[i])
            # Else if there is a stock price for a date a month and day after the app date, then store it
            elif bin_value_month_after_day_after != -1:
                closing_price_next.append(history[bin_value_month_after_day_after])
                date_next.append(next_dates_day_after[i])
            # Else we don't have data on any of the 3 dates so just add nans to the price and date list to merge with df
            else:
                closing_price_next.append(np.nan)
                date_next.append(np.nan)

        # Else add a nan because we don't have the data
        else:
            closing_price_current.append(np.nan)
            closing_price_next.append(np.nan)
            closing_price_prior.append(np.nan)
            date_prior.append(np.nan)
            date_current.append(np.nan)
            date_next.append(np.nan)

    # Add a nan if there was an error
    except:
        closing_price_current.append(np.nan)
        closing_price_next.append(np.nan)
        closing_price_prior.append(np.nan)
        date_prior.append(np.nan)
        date_current.append(np.nan)
        date_next.append(np.nan)

print("Price gathering progress: 100 % \n")

# Merges the lists with the original dataframe, drops na rows, and saves it
df['Month Before Application Date'] = date_prior
df['Application Date'] = date_current
df['Month After Application Date'] = date_next
df['Closing Price Last Month'] = closing_price_prior
df['Closing Price'] = closing_price_current
df['Closing Price Next Month'] = closing_price_next
df = df.dropna()
print("Current Inventor Dataset")
print(df)
df.to_csv('datasets/inventor-patent-tickers-dates-prices.csv')
