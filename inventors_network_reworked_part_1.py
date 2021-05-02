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

# Converts date to the correct format to get date a month before and after patent application date
for i in range(len(dates_list)):
    curr_date = dates_list[i]
    # Date format is not standardized in the dataset, so we chck for a / first
    if "/" in curr_date:
        try:
            d = datetime.datetime.strptime(curr_date, "%m/%d/%Y")
            curr_date = datetime.date.strftime(d, '%Y-%m-%d')
            curr_year = int(curr_date[:4])
            curr_month = int(curr_date[5:7])
            # processing to get dates one month before and after patent
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
            # Appends the new dates onto our lists
            curr_month = str(curr_month)
            curr_year = str(curr_year)
            app_date.append(curr_date)
            prev_date = curr_date.replace(curr_month, prev_month)
            prev_date = prev_date.replace(curr_year, prev_year)
            month_before_app_date.append(prev_date)
            next_date = curr_date.replace(curr_month, next_month)
            next_date = next_date.replace(curr_year, next_year)
            month_after_app_date.append(next_date)
        # If an error is thrown, we add nan to the list
        except:
            month_before_app_date.append(np.nan)
            app_date.append(np.nan)
            month_after_app_date.append(np.nan)
    # If it doesn't have a /, then we add / to the date and then process it
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
            # processing to get dates one month before and after patent
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
            # Appends the new dates onto our lists
            curr_month = str(curr_month)
            curr_year = str(curr_year)
            app_date.append(curr_date)
            prev_date = curr_date.replace(curr_month, prev_month)
            prev_date = prev_date.replace(curr_year, prev_year)
            month_before_app_date.append(prev_date)
            next_date = curr_date.replace(curr_month, next_month)
            next_date = next_date.replace(curr_year, next_year)
            month_after_app_date.append(next_date)
        # If an error is thrown, we add nan to the list
        except:
            month_before_app_date.append(np.nan)
            app_date.append(np.nan)
            month_after_app_date.append(np.nan)
        
# Creates a new dataframe to be used in Google Sheets to get stock prices
df["Month Before Application Date"] = month_before_app_date
df["Application Date"] = app_date
df["Month After Application Date"] = month_after_app_date
df.to_csv('tickers_and_dates_before_of_and_after.csv')

print(df)

