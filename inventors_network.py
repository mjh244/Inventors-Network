# Authors: Group 7 - McKenzie Hawkins, Alexander Mazon, Haolin Hu

# Imports
import csv
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

#####################################
# Network Creation
#####################################

# Creates the graphs
inventorNetwork = nx.Graph()
inventorNetworkSubset = nx.Graph()

# Gets values from the graph and puts them in their respective lists
firstNames = df["Firstname"].values
lastNames = df["Lastname"].values
inventors = []
for i in range(len(firstNames)):
    inventors.append(firstNames[i] + ' ' + lastNames[i])
patents = df["Patent"].values
companies = df["Assignee"].values

# Adds nodes and edges to the graph for the entire dataset
for i in range(len(companies)):

    # Adds nodes to the network if they weren't already in the network
    if (companies[i] not in inventorNetwork.nodes):
        inventorNetwork.add_node(companies[i], company = companies[i])
    if (patents[i] not in inventorNetwork):
        inventorNetwork.add_node(patents[i], patent = patents[i])
    if (inventors[i] not in inventorNetwork.nodes):
        inventorNetwork.add_node(inventors[i], inventor = inventors[i])

    # Adds edges to the network
    if (companies[i], patents[i]) not in inventorNetwork.edges:
        inventorNetwork.add_edge(companies[i], patents[i])
    if (patents[i], inventors[i]) not in inventorNetwork.edges:
        inventorNetwork.add_edge(patents[i], inventors[i])

# Adds nodes and edges to the graph for a subset of the dataset
for i in range(100):
    
    # Adds nodes to the network if they weren't already in the network
    if (companies[i] not in inventorNetworkSubset.nodes):
        inventorNetworkSubset.add_node(companies[i], company = companies[i], type = 'company')
    if (patents[i] not in inventorNetworkSubset):
        inventorNetworkSubset.add_node(patents[i], patent = patents[i], type = 'patent')
    if (inventors[i] not in inventorNetworkSubset.nodes):
        inventorNetworkSubset.add_node(inventors[i], inventor = inventors[i], type = 'inventor')

    # Adds edges to the network
    if (companies[i], patents[i]) not in inventorNetworkSubset.edges:
        inventorNetworkSubset.add_edge(companies[i], patents[i])
    if (patents[i], inventors[i]) not in inventorNetworkSubset.edges:
        inventorNetworkSubset.add_edge(patents[i], inventors[i])

# Displays the subset graph
nx.draw_networkx(inventorNetworkSubset, with_labels=True, font_size=10, node_size=100)
plt.title("Inventors Network (Subset)")
plt.show()


#####################################
# Metrics Calculation
#####################################

# Documentation on calculating metrics
# https://networkx.org/documentation/stable/reference/algorithms/centrality.html

# Documentation on getting nodes by their type/name
# https://networkx.org/documentation/networkx-2.1/reference/classes/generated/networkx.Graph.nodes.html

# Computes metrics for all nodes 
# (REMEMBER TO CHANGE BACK TO NORMAL, JUST USING SUBSET TO PRINT AND TEST)
# THIS IS THE EASIEST WAY BUT IDK HOW TO FILTER OUT COMPANIES ONLY FROM THIS

degree = nx.degree_centrality(inventorNetwork) 
closeness = nx.closeness_centrality(inventorNetwork) 
betweeness = nx.betweenness_centrality(inventorNetwork, normalized = True, endpoints = False) 
eigenvector = nx.eigenvector_centrality(inventorNetwork) 
"""
degree = nx.degree_centrality(inventorNetworkSubset) 
closeness = nx.closeness_centrality(inventorNetworkSubset) 
betweeness = nx.betweenness_centrality(inventorNetworkSubset, normalized = True, endpoints = False) 
eigenvector = nx.eigenvector_centrality(inventorNetworkSubset) 
"""
"""
print("---------------------Degree ------------------------------------")
print(degree)
print("---------------------Closeness ------------------------------------")
print(closeness)
print("---------------------Betweeness ------------------------------------")
print(betweeness)
print("---------------------Eigenvector ------------------------------------")
print(eigenvector)
"""

degreePerCompany = []
closenessPerCompany = []
betweenessPerCompany = []
eigenvectorPerCompany = []

# Gets the metrics based on company key and stores them in order of company to merge with df
for i in range(len(companies)):
#for i in range(100):
    degreePerCompany.append(degree.get(companies[i]))
    closenessPerCompany.append(closeness.get(companies[i]))
    betweenessPerCompany.append(betweeness.get(companies[i]))
    eigenvectorPerCompany.append(eigenvector.get(companies[i]))

#print(degreePerCompany)

# Adds the metrics to the dataframe
df['Degree'] = degreePerCompany
df['Closeness'] = closenessPerCompany
df['Betweeness'] = betweenessPerCompany
df['Eigenvector'] = eigenvectorPerCompany

print(len(companies))
print(len(degreePerCompany))
print(df)
