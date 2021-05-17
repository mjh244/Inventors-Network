# Authors: Group 7 - McKenzie Hawkins, Alexander Mazon, Haolin Hu

# Imports
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Reads in dataframe with stock prices
df = pd.read_csv('datasets/inventor-patent-tickers-dates-prices.csv')

# Drops an extra col that got added and drops na
df = df.drop(['Unnamed: 0'], axis=1)
df = df.dropna()

print(df)

# Renames columns from previous part to match this part
df = df.rename(columns={"Closing Price Last Month":"Price a Month Before"})
df = df.rename(columns={"Closing Price":"Price the Day of"})
df = df.rename(columns={"Closing Price Next Month":"Price a Month After"})

# Gets the dates from the df and stores them in a list as floats
last_month = df["Price a Month Before"].astype(float)
current_month = df["Price the Day of"].astype(float)
next_month = df["Price a Month After"].astype(float)

# Turns current col of assignees to a set to see number of unique companies
num_comp = df["Assignee"]
num_comp = set(num_comp)
print("Number of unique companies", len(num_comp))

# Computes the change col for labels in ML
# Shows whether price increased or decreased from app date to month after
# This is added to the dataframe at the end.
change = [next_month-current_month > 0]
change = np.array(change)
change = change.astype(int)
print(change[0])


#####################################
# Network Creation
#####################################

# Creates the graphs
inventorNetwork = nx.Graph()

# Gets the inventor names, patents, and assignees from the df and stores them in a list to use for network
firstNames = df["Firstname"].values
lastNames = df["Lastname"].values
inventors = []
for i in range(len(firstNames)):
    inventors.append(firstNames[i] + ' ' + lastNames[i])
patents = df["Patent"].values
companies = df["Assignee"].values

# We add nodes to the graph one at a time by looping through the new df
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
print("Degree calculated")
betweenness = nx.betweenness_centrality(inventorNetwork, 10, normalized = True, endpoints = False)
print("Betweenness calculated")
eigenvector = nx.eigenvector_centrality(inventorNetwork, max_iter=500, tol=0.0001)
print("Eigenvector calculated")

# Lists to hold company centrality metrics
degreePerInventor = []
betweennessPerInventor = []
eigenvectorPerInventor = []

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

# Drops price a month after because we dont want to train on that
df = df.drop(['Price a Month After'], axis=1)
# Drops app date because we made a standardized date of the one they give
df = df.drop(['AppDate'], axis=1)

print("Dataframe with metrics")
print(df)
df.to_csv('datasets/inventor-patent-tickers-dates-prices-centrality.csv')

# Gathers company names to gather nodes
allNodesByCompany = list(inventorNetwork.nodes)

# Sets node limit for subset network
nodeNames = []
nodeSubsetLimit = 8
for i in range(nodeSubsetLimit):
    nodeNames.append(allNodesByCompany[i])

# Creates subgraph of
inventorNetworkSubset = inventorNetwork.subgraph(nodeNames)


# Displays the subset graph
nx.draw_networkx(inventorNetworkSubset, with_labels=True, font_size=10, node_size=10)
plt.title("Inventors Network (Subset)")
plt.show()
plt.savefig('figures/inventor_network_subset.png')


# Turns features to numbers
for col in df.keys():
  if ((col == 'Firstname') or (col == 'Lastname') or (col == 'City') or (col == 'State') or (col == 'Country') 
  or (col == 'Zipcode') or (col == 'Patent') or (col == 'AppYear') or (col == 'GYear')  or (col == 'Assignee') 
  or (col == 'Tickers') or (col == 'Month Before Application Date') or (col == 'Application Date')
  or (col == 'Month After Application Date')):
    df[col] = pd.Categorical(df[col], ordered=True).codes
    df[col] = pd.to_numeric(df[col], downcast='float')

print(df)

# This applies standard scaling that data
scaled_df = StandardScaler().fit_transform(df)
df = pd.DataFrame(scaled_df)

# We now add the change column which indicates whether the stock increased or decreased in price.
df["Change"] = change[0]
df["Change"] = pd.to_numeric(df["Change"], downcast='float')

# Saves the dataframe with values converted to numbers to be used in ML
df.to_csv('datasets/inventor-patent-tickers-dates-prices-centrality-to-numbers.csv', index=False)
print("Dataset with values converted to numbers")
print(df)
