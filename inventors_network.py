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
df = pd.read_csv('invpat\invpat.csv')

# Removes unecessary field (WE MAY HAVE TO CHANGE IT TO OTHER FIELDS WE WANT TO REMOVE)
df = df.drop(['Street', 'Lat', 'Lng', 'InvSeq', 'AsgNum', 'Class', 'Invnum', 'Invnum_N', 'Invnum_N_UC', 'Density', 'Precision', 'Recall'], axis = 1)
df = df.dropna()

# Saves the filtered inventor dataframe as csv
df.to_csv('inventor-patent.csv') 

print(df)

#####################################
# Network Creation
#####################################

# Creates the graph
inventorNetwork = nx.Graph()

# Gets values from the graph and puts them in their respective lists
company = df["Assignee"].values
patent = df["Patent"].values
firstName = df["Firstname"].values
lastName = df["Lastname"].values
print(company)

# Adds nodes and edges to the graph
# Tried doing range(len(company)) but it wouldn't generate since its so large. NEED TO FIX
for i in range(100):
    inventorNetwork.add_node("Company: " + company[i])
    inventorNetwork.add_node("Patent: " + patent[i])
    inventorNetwork.add_edge("Company: " + company[i], "Patent: " + patent[i])
    inventorNetwork.add_node("Inventor: " + firstName[i] + ' ' + lastName[i])
    inventorNetwork.add_edge("Patent: " + patent[i], "Inventor: " + firstName[i] + ' ' + lastName[i])

# Displays the graph
nx.draw(inventorNetwork)
plt.show()



