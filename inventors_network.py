# Authors: Group 7 - McKenzie Hawkins, Alexander Mazon, Haolin Hu


# Imports
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yfinance as yf



######################################
# Data Manipulation
######################################

# Reads in the dataset to a dataframe
df = pd.read_csv('invpat.csv', dtype={'Zipcode': str, 'AsgNum': str, 'Invnum_N_UC': str})

# Removes unecessary field (WE MAY HAVE TO CHANGE IT TO OTHER FIELDS WE WANT TO REMOVE)
df = df.drop(['Street', 'Lat', 'Lng', 'InvSeq', 'AsgNum', 'Class', 'Invnum', 'Invnum_N', 'Invnum_N_UC', 'Density', 'Precision', 'Recall'], axis = 1)
df = df.dropna()
#set nodelimit early to make calculation and testing faster = 1000
df_starting_size = 30000
#we will not load the whole dataset for testing purposes, not sure if this speeds anything up so I commented it out
df = df.head(df_starting_size)

# Saves the filtered inventor dataframe as csv
df.to_csv('inventor-patent.csv')
#print(df)
#fulldf = df.copy()

######################################
# Stock Data Retrieval
######################################

# Ticker Retrieval Section

stocksDF = pd.read_csv('invpat/ticker_list.csv')

# Prints the dataframe containing various stock names and tickers
print(stocksDF)

# Set up array for ticker values
companyTickers = []

#assigns arrays to proper fields in the dataset
companyNames = df["Assignee"].tolist()
#companyNames = fulldf["Assignee"].tolist()
stockNames = stocksDF["Name"].tolist()
tickers = stocksDF["Ticker"].tolist()

# Gets the tickers of the companies from the new stocksDF
for i in range(len(df)):
    for j in range(len(stockNames)):
        # Checks if first word of assignee is equal to first word of stock name
        # Then adds the ticker to the list
        if (companyNames[i].split()[0] == stockNames[j].split()[0]):
            companyTickers.append(tickers[j])
            break
    # If nothing was added, do np.nan
    #the bound being <= is for when i=0, i dont think it chanegs behvaior for anything else.
    if (len(companyTickers) <= i):
        companyTickers.append(np.nan)



# Makes df only as long as tickers list
#df = df.replace(to_replace=None, value=np.nan).dropna()
#df = df.head(len(companyTickers))

# creates a new column in the df with the ticker values
df['Tickers'] = companyTickers
#drop rows from df with no ticker data
df = df.dropna()
####################################################################
print("THIS IS TICKERS")
print(tickers)
#this should have no NAN's in the ticker column
print("THIS IS DF")
print(df)

# Price Retrieval Section

ticker_list = df['Tickers'].to_numpy()
print("tickers follow")
print(ticker_list)
#trying to get histories for the tickers
hist_list=[]
for i in ticker_list:
    stock = yf.Ticker(i)
    hist = stock.history(period="max")
    hist_list.append(hist)
print("this is history of only the first ticker")
#an_history = hist_list[0]
an_history = hist_list[len(hist_list) - 1]
print(an_history)
print(an_history.keys())

print("historical info is of type" + str(type(hist_list[0])))

print("Length of list that holds ticker history")
print(len(hist_list))
print("Length of list that holds tickers")
print(len(ticker_list))

print("this is dataframe")
print(df)
closing_price_prior=[]
closing_price_current=[]
closing_price_next=[]

dates_list = df['AppDate'].to_numpy()

for i in range(len(dates_list)):
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
        if ((curr_date in history.index) and (next_date in history.index) and (prev_date in history.index)):
            closing_price_current.append(history.at[curr_date, 'Close'])
            closing_price_next.append(history.at[next_date, 'Close'])
            closing_price_prior.append(history.at[prev_date, 'Close'])
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
        if ((curr_date in history.index) and (next_date in history.index) and (prev_date in history.index)):
            closing_price_current.append(history.at[curr_date, 'Close'])
            closing_price_next.append(history.at[next_date, 'Close'])
            closing_price_prior.append(history.at[prev_date, 'Close'])
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



#####################################
# Network Creation
#####################################
# Creates the graphs
inventorNetwork = nx.Graph()

firstNames = df["Firstname"].values
lastNames = df["Lastname"].values
inventors = []
for i in range(len(firstNames)):
    inventors.append(firstNames[i] + ' ' + lastNames[i])
patents = df["Patent"].values
companies = df["Assignee"].values


#We add nodes to the graph one at a time by looping through the new df
for i in range(len(df)):

    # Adds nodes to the network if they weren't already in the network
    if (companies[i] not in inventorNetwork.nodes):
        inventorNetwork.add_node(companies[i], company = companies[i])
    if (patents[i] not in inventorNetwork.nodes):
        inventorNetwork.add_node(patents[i], patent = patents[i])
    if (inventors[i] not in inventorNetwork.nodes):
        inventorNetwork.add_node(inventors[i], inventor = inventors[i])

    # Adds edges to the network
    if (companies[i], patents[i]) not in inventorNetwork.edges:
        inventorNetwork.add_edge(companies[i], patents[i])
    if (patents[i], inventors[i]) not in inventorNetwork.edges:
        inventorNetwork.add_edge(patents[i], inventors[i])

#####################################
# Metrics Calculation
#####################################

# Computes metrics for nodes
degree = nx.degree_centrality(inventorNetwork)
print("degree calculated")
closeness = nx.closeness_centrality(inventorNetwork)
print("closeness calculated")
betweenness = nx.betweenness_centrality(inventorNetwork, 25, normalized = True, endpoints = False)
print("betweenness calculated")
eigenvector = nx.eigenvector_centrality(inventorNetwork, max_iter=1000)
print("eigenvector calculated")

# Lists to hold company centrality metrics
degreePerInventor = []
closenessPerInventor = []
betweennessPerInventor = []
eigenvectorPerInventor = []
inventors = list(inventors)

#just trying to figure out the bounds
print("inventors")
print(len(inventors))
print(len(degree))
print(len(closeness))
print(len(betweenness))
print(len(eigenvector))

# Gets the metrics based on company key and stores them in order of company to merge with df
#this was changed to loop through the length of companies so we dont get an out of bounds error
for i in range(len(inventors)):
    degreePerInventor.append(degree.get(inventors[i]))
    closenessPerInventor.append(closeness.get(inventors[i]))
    betweennessPerInventor.append(betweenness.get(inventors[i]))
    eigenvectorPerInventor.append(eigenvector.get(inventors[i]))

print("Finished computing lists \n")

# Limits df to number of companies centrality was calculated for
df = df.head(len(degree))

# Adds the metrics to the dataframe
df['Degree'] = degreePerInventor
df['Closeness'] = closenessPerInventor
df['Betweenness'] = betweennessPerInventor
df['Eigenvector'] = eigenvectorPerInventor

# Drops rows with empty fields and saves the df
df = df.dropna()
#dataframe with metrics shown
print("Dataframe with metrics")
print(df)
df.to_csv('inventor-patent-stock-centrality.csv')
# Gathers company names to gather nodes
allNodesByCompany = inventorNetwork.nodes(data = True)
allNodesByCompany2 = list(inventorNetwork.nodes)


#Sets node limit for subset network, I had to set it to 10 so it would be small
# enough given the initial size of our df was cut down by so much
nodeNames = []
nodeSubsetLimit = 10
for i in range(nodeSubsetLimit):
    nodeNames.append(allNodesByCompany2[i])

# Creates subgraph of
inventorNetworkSubset = inventorNetwork.subgraph(nodeNames)

# Displays the subset graph
nx.draw_networkx(inventorNetworkSubset, with_labels=True, font_size=10, node_size=100)
plt.title("Inventors Network (Subset)")
plt.show()

