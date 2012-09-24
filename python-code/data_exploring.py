"""
Kaggle Census competition
"""

import math

import numpy as np
import pandas 
import sklearn.linear_model as sklm
from sklearn.cross_validation import KFold

import census_utilities 

#try unpickling:
try:
    data = pandas.load( "census_data.pickle")
#open training_file
except:

    data = pandas.read_csv( "training_filev1.csv")
    data.save(  "census_data.pickle" )
    
print "Data in."
response = data['Mail_Return_Rate_CEN_2010']
weights = data['weight']





#preprocess data.
data = census_utilities.preprocess_dataframe( data )
"Data cleaned. TODO: check cleaning algos"
#Ok data is pretty clean, what can we do with it?

n, d = data.shape




print help(census_utilities.cv)








