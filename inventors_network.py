# -*- coding: utf-8 -*-
"""Inventors_Network.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1I6J93OI0TahODpG4Utsg5z7puUpHZYVZ
"""

# Authors: McKenzie Hawkins, Alexander Mazon, Haolin Hu

# Imports
import csv
import pandas as pd
import numpy as np
from linkedin_scraper import Person, actions
from selenium import webdriver

######################################
# Data Manipulation
######################################

# Reads in the dataset to a dataframe
df = pd.read_csv('invpat\invpat.csv')

# Removes unecessary field (WE MAY HAVE TO CHANGE IT TO OTHER FIELDS WE WANT TO REMOVE)
df = df.drop(['Street', 'City', 'State', 'Country', 'Zipcode', 'Lat', 'Lng', 'Class', 'Invnum_N_UC', 'Density', 'Precision', 'Recall'], axis = 1)
df = df.dropna()

# Saves the filtered inventor dataframe as csv
df.to_csv('inventor-patent.csv') 

print(df)

#####################################
# Inventor Data Retrieval
#####################################

# This should log us into our account and get us our info (havn't tested yet).
email = "st3407126@protonmail.com"
password = "csdsgroup7"
actions.login(driver, email, password) 
person = Person("https://www.linkedin.com/in/student-testaccount-4742b9208", driver=driver)