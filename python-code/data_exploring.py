"""
Kaggle Census competition
"""

import math

import numpy as np
import pandas 
import sklearn.linear_model as sklm
from sklearn.cross_validation import KFold

import census_utilities 




training_file = "census_data_sample.csv"
#try unpickling:
try:
    data = pandas.load( "../"+training_file.split(".")[0]+".pickle")
#open training_file
except:

    data = pandas.read_csv( "../"+training_file)
    data.save(  "../"+training_file.split(".")[0]+".pickle" )
    
print "Data in."
response = data['Mail_Return_Rate_CEN_2010']
weights = data['weight']





#preprocess data.
data = census_utilities.preprocess_dataframe( data )
"Data cleaned. TODO: check cleaning algos"
#Ok data is pretty clean, what can we do with it?

n, d = data.shape



print help(census_utilities.cv)








