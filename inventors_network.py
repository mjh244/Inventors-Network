# Authors: Group 7 - McKenzie Hawkins, Alexander Mazon, Haolin Hu

# Imports
import csv
import pandas as pd
import numpy as np
import igraph
import networkx as nx
import matplotlib.pyplot as plt
import yfinance as yf

######################################
# Data Manipulation
######################################

# Reads in the dataset to a dataframe
df = pd.read_csv('invpat/invpat.csv', dtype={'Zipcode': str, 'AsgNum': str, 'Invnum_N_UC': str})

# Removes unecessary field (WE MAY HAVE TO CHANGE IT TO OTHER FIELDS WE WANT TO REMOVE)
df = df.drop(['Street', 'Lat', 'Lng', 'InvSeq', 'AsgNum', 'Class', 'Invnum', 'Invnum_N', 'Invnum_N_UC', 'Density', 'Precision', 'Recall'], axis = 1)
df = df.dropna()

# Saves the filtered inventor dataframe as csv
df.to_csv('inventor-patent.csv')
print(df)
fulldf = df.copy()

#####################################
# Network Creation
#####################################

# Creates the graphs
inventorNetwork = nx.Graph()

# Gets values from the graph and puts them in their respective lists
firstNames = df["Firstname"].values
lastNames = df["Lastname"].values
inventors = []
for i in range(len(firstNames)):
    inventors.append(firstNames[i] + ' ' + lastNames[i])
patents = df["Patent"].values
companies = df["Assignee"].values

# Sets node limit for network and adds nodes and edges to the graph
nodeLimit = 10000
for i in range(nodeLimit):

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


# Gathers company names to gather nodes
allNodesByCompany = inventorNetwork.nodes(data = True)
allNodesByCompany2 = list(inventorNetwork.nodes)
nodeNames = []

#Sets node limit for subset network
nodeSubsetLimit = 100
for i in range(nodeSubsetLimit):
    nodeNames.append(allNodesByCompany2[i])

# Creates subgraph of 
inventorNetworkSubset = inventorNetwork.subgraph(nodeNames)

# Displays the subset graph
nx.draw_networkx(inventorNetworkSubset, with_labels=True, font_size=10, node_size=100)
plt.title("Inventors Network (Subset)")
plt.show()

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
eigenvector = nx.eigenvector_centrality(inventorNetwork) 
print("eigenvector calculated")

# Lists to hold company centrality metrics
degreePerCompany = []
closenessPerCompany = []
betweennessPerCompany = []
eigenvectorPerCompany = []

# Gets the metrics based on company key and stores them in order of company to merge with df
for i in range(len(degree)):
    degreePerCompany.append(degree.get(companies[i]))
    closenessPerCompany.append(closeness.get(companies[i]))
    betweennessPerCompany.append(betweenness.get(companies[i]))
    eigenvectorPerCompany.append(eigenvector.get(companies[i]))

print("Finished computing lists \n")

# Limits df to number of companies centrality was calculated for
df = df.head(len(degree))

# Adds the metrics to the dataframe
df['Degree'] = degreePerCompany
df['Closeness'] = closenessPerCompany
df['Betweeness'] = betweennessPerCompany
df['Eigenvector'] = eigenvectorPerCompany

# Drops rows with empty fields and saves the df
df = df.dropna()
print(df)
df.to_csv('inventor-patent-ten-thousand-nodes.csv')

#####################################
# Stock Market Data Retrieval
#####################################

# Obtained csv of tickers and company names from the following link
# https://www.quandl.com/data/EOD-End-of-Day-US-Stock-Prices/documentation

# Useful Links
# https://www.quandl.com/data/EOD-End-of-Day-US-Stock-Prices/documentation
# https://stackoverflow.com/questions/32383585/turn-list-of-company-names-into-tickers
# https://www.nasdaq.com/market-activity/stocks/screener
# https://stackoverflow.com/questions/38967533/retrieve-company-name-with-ticker-symbol-input-yahoo-or-google-api
# 
# Reads in the stock ticker and name dataset to a dataframe
stocksDF = pd.read_csv('invpat/ticker_list.csv')

# Prints the dataframe containing various stock names and tickers
print(stocksDF)

companyTickers = []

companyNames = df["Assignee"].tolist()
#companyNames = fulldf["Assignee"].tolist()
stockNames = stocksDF["Name"].tolist()
tickers = stocksDF["Ticker"].tolist()


for i in range(len(companyNames)):
    companyNames[i] = companyNames[i].split()[0]

for i in range(len(stockNames)):
    stockNames[i] = stockNames[i].split()[0]

# Gets the tickers of the companies from the new stocksDF
for i in range(len(df)):
    for j in range(len(stockNames)):
        # Checks if first word of assignee is equal to first word of stock name
        # Then adds the ticker to the list
        if (companyNames[i] == stockNames[j]):
            companyTickers.append(tickers[j])
            break
    # If nothing was added, add a None. This way when we merge the tickets with the df,
    # we can take out rows that don't correspond to a ticket.
    if (len(companyTickers) != i):
        companyTickers.append(None)

# Makes df only as long as tickers list
df = df.head(len(companyTickers))
df['Tickers'] = companyTickers

# Drops rows where tickers couldn't be matched
df = df.replace(to_replace='None', value=np.nan).dropna()
print(df)

