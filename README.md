# Inventors-Network
#This is a repo for our class project in our Senior Project class. This repo contains code we used to create an inventors network to predict stock performance.

#The code for the project is contained within the inventors_network.py file.

#Required Files:
- The inventor dataset is required before running the code. This file is called invpat.zip and it is from from Harvard’s “Disambiguation and Co-authorship Networks of the U.S. Patent Inventor Database (1975 - 2010)”. This file was not uploaded to the repo as it exceeded Github's 100 MB capacity. It can be found at the following link and should be saved in the same location as the inventors_network.py file, as well as the ticker-list.csv file as it too is used in the code.
https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/5F1RRI/OR52T1&version=5.1


# You will also need to install the following libraries:
- pandas
- datetime
- matplotlib.pyplot
- networkx
- numpy
- yfinance
- sklearn

# Currently our program is split into 4 parts labeled as follows:
- inventors_network_reworked_part_1.py, which contains the code for taking in the dataset and matching assignee names with tickers from the Quandl dataset. The dates for a month before, and a month after the aptent application date are also computed and appended to the end of the dataframe, which is then save as a csv.
- Stock Retrieval, which is not a file, but rather a stage where we uploaded our new dataset from part 1 to Google Sheets. Then prices for the tickers combined with the dates were retrieved using Google Sheets built in Google Finance library.
- inventors_network_reworked_part_2.py, which contains the code for taking in the new dataset with prices from Google Sheets, and creates a network based on the inventors, assignees, and patents. The resulting file is a csv that contains the degree, betweenness, and eigenvector centrality measures appended to the previous dataset.
- inventors_network_reworked_part_3.py, which contains the code for performing the machine learning on the dataset from part 2. This is where the results of machine learning is shown.