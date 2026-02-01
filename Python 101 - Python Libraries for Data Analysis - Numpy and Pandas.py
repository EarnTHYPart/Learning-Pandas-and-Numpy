#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python 101: Python Libraries for Data Analysis - NumPy and Pandas
==================================================================

This tutorial covers fundamental concepts and operations for:
- NumPy: Linear algebra library for multidimensional arrays
- Pandas: Data manipulation and analysis tool built on NumPy

Author: Learning Pandas and Numpy
Date: 2026
"""

import numpy as np
import pandas as pd

# TASK #1: DEFINE SINGLE AND MULTI-DIMENSIONAL NUMPY ARRAYS
# NumPy is a Linear Algebra Library used for multidimensional arrays
# NumPy brings the best of two worlds: (1) C/Fortran computational efficiency,
# (2) Python language easy syntax

# Define a one-dimensional array
my_list = [1, 2, 3, 4, 5]
arr_1d = np.array(my_list)
print("1D Array:", arr_1d)



# Multi-dimensional array (Matrix definition)
arr_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print("2D Array:\n", arr_2d)

# MINI CHALLENGE #1:
# - Write a code that creates the following 2x4 numpy array:
# [[3 7 9 3]
#  [4 3 2 2]]
# TODO: Write your solution below


# TASK #2: LEVERAGE NUMPY BUILT-IN METHODS AND FUNCTIONS

# Uniform distribution between 0 and 1 using rand()
rand_array = np.random.rand(3, 3)
print("Random array with rand():\n", rand_array)

# Create a matrix of random numbers
rand_matrix = np.random.rand(2, 4)
print("Random matrix:\n", rand_matrix)

# Generate random integers between upper and lower bounds using randint()
rand_integers = np.random.randint(1, 10, size=5)
print("Random integers:", rand_integers)

# Generate a specific number of random integers
rand_integers_array = np.random.randint(1, 100, 10)
print("Array of random integers:", rand_integers_array)

# Create evenly spaced values using arange()
arange_array = np.arange(0, 10, 2)
print("Array with arange():", arange_array)

# Create identity matrix (diagonal of ones, zeros everywhere else)
identity = np.eye(3)
print("Identity matrix:\n", identity)

# Matrix of ones
ones_matrix = np.ones((3, 3))
print("Matrix of ones:\n", ones_matrix)

# Array of zeros
zeros_array = np.zeros(5)
print("Array of zeros:", zeros_array)

# MINI CHALLENGE #2:
# - Write a code that takes in a positive integer "x" from the user
#   and creates a 1x10 array with random numbers ranging from 0 to "x"
# TODO: Write your solution below





# TASK #3: PERFORM MATHEMATICAL OPERATIONS IN NUMPY


# np.arange() returns an evenly spaced values within a given interval


# In[17]:





# In[18]:


# Add 2 numpy arrays together


# In[19]:





# In[20]:





# In[21]:





# MINI CHALLENGE #3:
# - Given the X and Y values below, obtain the distance between them
# 
# ```
# X = [5, 7, 20]
# Y = [9, 15, 4]
# ```

# In[ ]:





# TASK #4: PERFORM ARRAY SLICING AND INDEXING





# In[27]:


# Access specific index from the numpy array


# In[28]:


# Starting from the first index 0 up until and NOT including the last element


# In[29]:


# Broadcasting, altering several values in a numpy array at once


# In[30]:


# Let's define a two dimensional numpy array


# In[31]:


# Get a row from a mtrix


# In[32]:


# Get one element


# MINI CHALLENGE #4:
# - In the following matrix, replace the last row with 0
# 
# ```
# X = [2 30 20 -2 -4]
#     [3 4  40 -3 -2]
#     [-3 4 -6 90 10]
#     [25 45 34 22 12]
#     [13 24 22 32 37]
# ```
# 
# 

# In[ ]:





# TASK #5: PERFORM ELEMENTS SELECTION (CONDITIONAL)





# In[43]:





# In[44]:


# Obtain odd elements only


# MINI CHALLENGE #5:
# - In the following matrix, replace negative elements by 0 and replace odd elements with -2
# 
# 
# ```
# X = [2 30 20 -2 -4]
#     [3 4  40 -3 -2]
#     [-3 4 -6 90 10]
#     [25 45 34 22 12]
#     [13 24 22 32 37]
# ```
# 

# In[ ]:





# TASK #6: UNDERSTAND PANDAS FUNDAMENTALS


# Pandas is a data manipulation and analysis tool that is built on Numpy.
# Pandas uses a data structure known as DataFrame (think of it as Microsoft excel in Python). 
# DataFrames empower programmers to store and manipulate data in a tabular fashion (rows and columns).
# Series Vs. DataFrame? Series is considered a single column of a DataFrame.


# In[45]:





# In[53]:


# Let's define a two-dimensional Pandas DataFrame
# Note that you can create a pandas dataframe from a python dictionary


# In[54]:


# Let's obtain the data type 


# In[55]:


# you can only view the first couple of rows using .head()


# In[56]:


# you can only view the last couple of rows using .tail()


# MINI CHALLENGE #6:
# - A porfolio contains a collection of securities such as stocks, bonds and ETFs. Define a dataframe named 'portfolio_df' that holds 3 different stock ticker symbols, number of shares, and price per share (feel free to choose any stocks)
# - Calculate the total value of the porfolio including all stocks

# In[ ]:





# TASK #7: PANDAS WITH CSV AND HTML DATA


# Pandas is used to read a csv file and store data in a DataFrame


# In[63]:





# In[64]:


# Read tabular data using read_html


# In[65]:





# In[66]:





# MINI CHALLENGE #7:
# - Write a code that uses Pandas to read tabular US retirement data
# - You can use data from here: https://www.ssa.gov/oact/progdata/nra.html 

# In[ ]:





# TASK #8: PANDAS OPERATIONS


# Let's define a dataframe as follows:


# In[68]:


# Pick certain rows that satisfy a certain criteria 


# In[69]:


# Delete a column from a DataFrame


# MINI CHALLENGE #8:
# - Using "bank_client_df" DataFrame, leverage pandas operations to only select high networth individuals with minimum $5000 
# - What is the combined networth for all customers with 5000+ networth?

# In[ ]:





# TASK #9: PANDAS WITH FUNCTIONS


# Let's define a dataframe as follows:
bank_client_df = pd.DataFrame({'Bank client ID':[111, 222, 333, 444], 
                               'Bank Client Name':['Chanel', 'Steve', 'Mitch', 'Ryan'], 
                               'Net worth [$]':[3500, 29000, 10000, 2000], 
                               'Years with bank':[3, 4, 9, 5]})
bank_client_df


# In[73]:


# Define a function that increases all clients networth (stocks) by a fixed value of 20% (for simplicity sake) 


# In[74]:


# You can apply a function to the DataFrame 


# In[75]:





# MINI CHALLENGE #9:
# - Define a function that triples the stock prices and adds $200
# - Apply the function to the DataFrame
# - Calculate the updated total networth of all clients combined

# In[ ]:





# TASK #10: PERFORM SORTING AND ORDERING IN PANDAS


# Let's define a dataframe as follows:
bank_client_df = pd.DataFrame({'Bank client ID':[111, 222, 333, 444], 
                               'Bank Client Name':['Chanel', 'Steve', 'Mitch', 'Ryan'], 
                               'Net worth [$]':[3500, 29000, 10000, 2000], 
                               'Years with bank':[3, 4, 9, 5]})
bank_client_df


# In[81]:


# You can sort the values in the dataframe according to number of years with bank


# In[82]:


# Note that nothing changed in memory! you have to make sure that inplace is set to True


# In[83]:


# Set inplace = True to ensure that change has taken place in memory 


# In[84]:


# Note that now the change (ordering) took place 


# TASK #11: PERFORM CONCATENATING AND MERGING WITH PANDAS


# Check this out: https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html


# In[91]:





# In[92]:





# In[93]:





# In[94]:





# In[95]:





# In[96]:





# In[97]:





# TASK #12: PROJECT AND CONCLUDING REMARKS named 'Bank_df_1' that contains the first and last names for 5 bank clients with IDs = 1, 2, 3, 4, 5 
# - Assume that the bank got 5 new clients, define another dataframe named 'Bank_df_2' that contains a new clients with IDs = 6, 7, 8, 9, 10
# - Let's assume we obtained additional information (Annual Salary) about all our bank customers (10 customers) 
# - Concatenate both 'bank_df_1' and 'bank_df_2' dataframes
# - Merge client names and their newly added salary information using the 'Bank Client ID'
# - Let's assume that you became a new client to the bank
# - Define a new DataFrame that contains your information such as client ID (choose 11), first name, last name, and annual salary.
# - Add this new dataframe to the original dataframe 'bank_df_all'.

# In[ ]:





# # EXCELLENT JOB!

# # MINI CHALLENGES SOLUTIONS

# MINI CHALLENGE #1 SOLUTION: 
# - Write a code that creates the following 2x4 numpy array
# 
# ```
# [[3 7 9 3] 
# [4 3 2 2]]
# ```

# In[5]:


x = np.array([[[3, 7, 9, 3] , [4, 3, 2, 2]]])
x


# MINI CHALLENGE #2 SOLUTION:
# - Write a code that takes in a positive integer "x" from the user and creates a 1x10 array with random numbers ranging from 0 to "x"

# In[15]:


x = int(input("Please enter a positive integer value: "))
x = np.random.randint(1, x, 10)
x


# In[ ]:





# MINI CHALLENGE #3 SOLUTION:
# - Given the X and Y values below, obtain the distance between them
# 
# 
# ```
# X = [5, 7, 20]
# Y = [9, 15, 4]
# ```
# 
# 
# 

# In[22]:


X = np.array([5, 7, 20])
Y = np.array([9, 15, 4])
Z = np.sqrt(X**2 + Y**2)
Z


# MINI CHALLENGE #4 SOLUTION:
# - In the following matrix, replace the last row with 0
# 
# ```
# X = [2 30 20 -2 -4]
#     [3 4  40 -3 -2]
#     [-3 4 -6 90 10]
#     [25 45 34 22 12]
#     [13 24 22 32 37]
# ```
# 
# 
# 

# In[33]:


X = np.array([[2, 30, 20, -2, -4],
    [3, 4,  40, -3, -2],
    [-3, 4, -6, 90, 10],
    [25, 45, 34, 22, 12],
    [13, 24, 22, 32, 37]])


# In[34]:


X[4] = 0
X


# MINI CHALLENGE #5 SOLUTION:
# - In the following matrix, replace negative elements by 0 and replace odd elements with -2
# 
# 
# ```
# X = [2 30 20 -2 -4]
#     [3 4  40 -3 -2]
#     [-3 4 -6 90 10]
#     [25 45 34 22 12]
#     [13 24 22 32 37]
# ```

# In[41]:


X = np.array([[2, 30, 20, -2, -4],
    [3, 4,  40, -3, -2],
    [-3, 4, -6, 90, 10],
    [25, 45, 34, 22, 12],
    [13, 24, 22, 32, 37]])

X[X<0] = 0
X[X%2==1] = -2
X


# MINI CHALLENGE #6 SOLUTION:
# - A porfolio contains a collection of securities such as stocks, bonds and ETFs. Define a dataframe named 'portfolio_df' that holds 3 different stock ticker symbols, number of shares, and price per share (feel free to choose any stocks)
# - Calculate the total value of the porfolio including all stocks

# In[58]:


portfolio_df = pd.DataFrame({'stock ticker symbols':['AAPL', 'AMZN', 'T'],
                             'price per share [$]':[3500, 200, 40], 
                             'Number of stocks':[3, 4, 9]})
portfolio_df


# In[59]:


stocks_dollar_value = portfolio_df['price per share [$]'] * portfolio_df['Number of stocks']
print(stocks_dollar_value)
print('Total portfolio value = {}'.format(stocks_dollar_value.sum()))


# MINI CHALLENGE #7 SOLUTION:
# - Write a code that uses Pandas to read tabular US retirement data
# - You can use data from here: https://www.ssa.gov/oact/progdata/nra.html 

# In[ ]:


# Read tabular data using read_html
retirement_age_df = pd.read_html('https://www.ssa.gov/oact/progdata/nra.html')
retirement_age_df


# MINI CHALLENGE #8 SOLUTION:
# - Using "bank_client_df" DataFrame, leverage pandas operations to only select high networth individuals with minimum $5000 
# - What is the combined networth for all customers with 5000+ networth?

# In[ ]:


df_high_networth = bank_client_df[ (bank_client_df['Net worth [$]'] >= 5000) ]
df_high_networth


# In[ ]:


df_high_networth['Net worth [$]'].sum()


# MINI CHALLENGE #9 SOLUTION:
# - Define a function that triples the stock prices and adds $200
# - Apply the function to the DataFrame
# - Calculate the updated total networth of all clients combined

# In[77]:


def networth_update(balance):
    return balance * 3 + 200 


# In[78]:


# You can apply a function to the DataFrame 
results = bank_client_df['Net worth [$]'].apply(networth_update)
results


# In[79]:


results.sum()


# PROJECT SOLUTION:

# In[ ]:


# Creating a dataframe from a dictionary
# Let's define a dataframe with a list of bank clients with IDs = 1, 2, 3, 4, 5 

raw_data = {'Bank Client ID': ['1', '2', '3', '4', '5'],
            'First Name': ['Nancy', 'Alex', 'Shep', 'Max', 'Allen'], 
            'Last Name': ['Rob', 'Ali', 'George', 'Mitch', 'Steve']}

Bank_df_1 = pd.DataFrame(raw_data, columns = ['Bank Client ID', 'First Name', 'Last Name'])
Bank_df_1


# Let's define another dataframe for a separate list of clients (IDs = 6, 7, 8, 9, 10)
raw_data = {
        'Bank Client ID': ['6', '7', '8', '9', '10'],
        'First Name': ['Bill', 'Dina', 'Sarah', 'Heather', 'Holly'], 
        'Last Name': ['Christian', 'Mo', 'Steve', 'Bob', 'Michelle']}
Bank_df_2 = pd.DataFrame(raw_data, columns = ['Bank Client ID', 'First Name', 'Last Name'])
Bank_df_2


# Let's assume we obtained additional information (Annual Salary) about our bank customers 
# Note that data obtained is for all clients with IDs 1 to 10 
raw_data = {
        'Bank Client ID': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        'Annual Salary [$/year]': [25000, 35000, 45000, 48000, 49000, 32000, 33000, 34000, 23000, 22000]}
bank_df_salary = pd.DataFrame(raw_data, columns = ['Bank Client ID','Annual Salary [$/year]'])
bank_df_salary


# Let's concatenate both dataframes #1 and #2
# Note that we now have client IDs from 1 to 10
bank_df_all = pd.concat([Bank_df_1, Bank_df_2])
bank_df_all


# Let's merge all data on 'Bank Client ID'
bank_df_all = pd.merge(bank_df_all, bank_df_salary, on = 'Bank Client ID')
bank_df_all


# In[104]:


new_client = {
        'Bank Client ID': ['11'],
        'First Name': ['Ry'], 
        'Last Name': ['Aly'],
        'Annual Salary [$/year]' : [1000]}
new_client_df = pd.DataFrame(new_client, columns = ['Bank Client ID', 'First Name', 'Last Name', 'Annual Salary [$/year]'])
new_client_df


# In[105]:


new_df = pd.concat([bank_df_all, new_client_df], axis = 0)
new_df

