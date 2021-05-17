# Authors (Group 7)
- McKenzie Hawkins
- Alexander Mazon
- Haolin Hu

# Inventors Network
This is a repo for our class project in our Senior Project in Computer Science class. This repo contains code and resulting dataset that we used to create a network of inventors to predict the stock performance of the companies that the inventors worked at.

# Required Files:
- The inventor dataset is required before running the code. This file is called invpat.zip and it is from from Harvard’s “Disambiguation and Co-authorship Networks of the U.S. Patent Inventor Database (1975 - 2010)”. This file was not uploaded to the repo as it exceeded Github's 100 MB capacity. It can be found at the following link and should be saved in the same location as the inventors_network.py file, as well as the ticker-list.csv file as it too is used in the code.
https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/5F1RRI/OR52T1&version=5.1


# You may also need to install the following libraries:
- pandas
    - How to install with pip: `pip install pandas`
- numpy
    - How to install with pip: `pip install numpy`
- datetime
    - How to install with pip: `pip install DateTime`
- bisect
    - Should be a built in Python module
- dateutil.relativedelta
    - How to install with pip: `pip install python-dateutil`
- csv
    - Should be a built in Python module
- matplotlib.pyplot
    - How to install with pip: `pip install matplotlib`
- networkx
    - How to install with pip: `pip install networkx`
- yfinance
    - How to install with pip: `pip install yfinance`
- sklearn
    - How to install with pip: `pip install -U scikit-learn`

# Our program is split into 3 parts and they are labeled as follows:
- inventors_network_reworked_part_1.py, which contains the code for taking in the dataset and matching assignee names with tickers from the Quandl dataset. The dates for a month before, and a month after the patent application date are also computed and appended to the end of the dataframe. Then the stock prices for the companies are retrieved using yfinance and appended to the end of the dataframe, which is then saved as a csv.
    - Estimated Runtime: ~15 minutes mostly due to downloading stock data
- inventors_network_reworked_part_2.py, which creates a network based on the inventors, assignees, and patents. There are two resulting files that are both csv's that contain the degree, betweenness, and eigenvector centrality measures appended to the previous dataset. One dataset has each value converted to a number to be used in the machine learning, while the other does not.
    - Estimated Runtime: ~3 minutes
- inventors_network_reworked_part_3.py, which contains the code for performing the machine learning on the dataset from part 2. This is where the testing, validation, and results of machine learning for our best model is shown. Currently we are using a decision tree model that is getting approximately 63% accuracy.
    - Estimated Runtime: ~30 minutes for full testing, validation, and parameter tuning.
- inventors_network_ANN.py, which contains our Artificial Neural Network using the Keras library. This is where we perform the the testing, validation, and prediction for the ANN. Our accuracy for the ANN was about 55%, which is about where most of our other models get as well,  other than the decision tree
    - Estimated Runtime: ~30 minutes for full testing, validation, and parameter tuning.
- inventors_network_ongoing_analysis.py, which contains the code for testing different machine learning models on the dataset from part 2. This is where the results of machine learning for a variety of our other models such as Naive Bayes and Random Forest. We tested models in this code file by commenting out all other sections of code in this file to test our desired models. Currently every model is uncommented in this file, so if you want to run it, we suggest commenting out all models but one to save on runtime.
    - Estimated Runtime: ~30 minutes for each model using full testing, validation, and parameter tuning. 

# Commit notes
-   GitHub shows that McKenzie Hawkins committed and deleted over 5 million lines of code, but we believe that those numbers that is mostly from uplaoding and deleting our dataset, as at one point in time after our preprocessing we ended up with a much lower dataset than the one that we currently use, so much lower that we were able to upload it to the repo. That is not the case anymore due to changes in our preprocessing, so those datasets had to be deleted from the repo. This is likely the reason as to why it shows so many lines were added and removed.
-   Haolin Hu did commit to the repo as can be seen in the commit history, but for some reason github is not listing him as a contributor and we do not know why.