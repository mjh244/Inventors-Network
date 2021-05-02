# Authors: Group 7 - McKenzie Hawkins, Alexander Mazon, Haolin Hu

# Imports
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Reads in dataframe with stock prices obtained from Google Sheets
#df = pd.read_csv('inventor-patent-tickers-dates-prices-full.csv')
df = pd.read_csv('inventor-patent-tickers-prices-full.csv')
df = df.drop(['Unnamed: 0'], axis=1)
df = df.dropna()
print(len(df))
print(df)
print(len(df))

df = df.rename(columns={"Closing Price Last Month":"Price a Month Before"})
df = df.rename(columns={"Closing Price":"Price the Day of"})
df = df.rename(columns={"Closing Price Next Month":"Price a Month After"})

last_month = df["Price a Month Before"].astype(float)
current_month = df["Price the Day of"].astype(float)
next_month = df["Price a Month After"].astype(float)

test_set = df["Assignee"]
test_set = set(test_set)
print("NUMBER OF COMPANIES")
print(len(test_set))


change = [next_month-current_month > 0]
change = np.array(change)
change = change.astype(int)
print(change[0])

df["Change"] = change[0]

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
betweenness = nx.betweenness_centrality(inventorNetwork, 25, normalized = True, endpoints = False)
print("betweenness calculated")
eigenvector = nx.eigenvector_centrality(inventorNetwork, max_iter=1000)
print("eigenvector calculated")

# Lists to hold company centrality metrics
degreePerInventor = []
betweennessPerInventor = []
eigenvectorPerInventor = []
inventors = list(inventors)

# Gets the metrics based on company key and stores them in order of company to merge with df
for i in range(len(inventors)):
    degreePerInventor.append(degree.get(inventors[i]))
    betweennessPerInventor.append(betweenness.get(inventors[i]))
    eigenvectorPerInventor.append(eigenvector.get(inventors[i]))

print("Finished computing lists \n")

# Adds the metrics to the dataframe
df['Degree'] = degreePerInventor
df['Betweenness'] = betweennessPerInventor
df['Eigenvector'] = eigenvectorPerInventor

# Drops rows with empty fields and saves the df
df = df.dropna()

print("Dataframe with metrics")
print(df)
df.to_csv('inventor-patent-stock-centrality.csv')

# Gathers company names to gather nodes
allNodesByCompany = inventorNetwork.nodes(data = True)
allNodesByCompany2 = list(inventorNetwork.nodes)

# Sets node limit for subset network
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


str_cols = df.keys()
str_cols = str_cols[1:]
print(str_cols)

# Map nominal features to numbers
for col in str_cols:
  if ((col == 'Firstname') or (col == 'Lastname') or (col == 'City') or (col == 'State') or (col == 'Country') 
  or (col == 'Zipcode') or (col == 'Patent') or (col == 'AppYear') or (col == 'GYear') or (col == 'AppDate') 
  or (col == 'Assignee') or (col == 'Tickers') or (col == 'Month Before Application Date') or (col == 'Application Date')
  or (col == 'Month After Application Date')):
    df[col] = pd.Categorical(df[col], ordered=True).codes
    pd.to_numeric(df[col], downcast='float')
#df = df.apply(pd.to_numeric)
#print(df)

df = df.drop(['Price a Month After'], axis=1)
df = df.drop(['AppDate'], axis=1)


df.to_csv('inventor-patent-tickers-dates-prices-centrality-to-numbers-full.csv', index=False)
print("PRINTING")
print(df)

