# Authors (Group 7)
- McKenzie Hawkins
- Alexander Mazon
- Haolin Hu

# Inventors-Network
This is a repo for our class project in our Senior Project class. This repo contains code we used to create a network of inventors to predict stock performance.

The code for the project is contained within the inventors_network.py file.

# Required Files:
- The inventor dataset is required before running the code. This file is called invpat.zip and it is from from Harvard’s “Disambiguation and Co-authorship Networks of the U.S. Patent Inventor Database (1975 - 2010)”. This file was not uploaded to the repo as it exceeded Github's 100 MB capacity. It can be found at the following link and should be saved in the same location as the inventors_network.py file, as well as the ticker-list.csv file as it too is used in the code.
https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/5F1RRI/OR52T1&version=5.1


# You will also need to install the following libraries:
- pandas
- numpy
- datetime
- bisect
- dateutil.relativedelta
- csv
- matplotlib.pyplot
- networkx
- yfinance
- sklearn

# Uur program is split into 3 parts and they are labeled as follows:
- inventors_network_reworked_part_1.py, which contains the code for taking in the dataset and matching assignee names with tickers from the Quandl dataset. The dates for a month before, and a month after the patent application date are also computed and appended to the end of the dataframe. Then the stock prices for the companies are retrieved using yfinance and appended to the end of the dataframe, which is then save as a csv.
-- Estimated Runtime: ~15 minutes mostly due to downloading stock data
- inventors_network_reworked_part_2.py, which contains the code for taking in the new dataset with prices from Google Sheets, and creates a network based on the inventors, assignees, and patents. There are two resulting files that are both csv's that contain the degree, betweenness, and eigenvector centrality measures appended to the previous dataset. One file has each value converted to a number to be used in the machine learning, the other does not.
-- Estimated Runtime: ~2 minutes
- inventors_network_reworked_part_3.py, which contains the code for performing the machine learning on the dataset from part 2. This is where the results of machine learning is shown. Currently we are using a decision tree model that is getting approximately 69% accuracy.
-- Estimated Runtime: ~5 minutes