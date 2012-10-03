"""
Kaggle Census competition
"""

import math

import numpy as np
import pandas 
import sklearn.linear_model as sklm
from census_utilities import *

import geoNN
import localRegression as lclR


training_file = "TrainingData.csv"
#try unpickling:
try:
    data = pandas.load( "../"+training_file.split(".")[0]+".pickle")
#open training_file
except:

    data = pandas.read_csv( "../"+training_file)
    data.save(  "../"+training_file.split(".")[0]+".pickle" )
    
print "Data in."






#del data['Mail_Return_Rate_CEN_2010']
#del data['weight']
#del data[ data.columns[0] ]
#preprocess data.
data, location_data, response, weights = preprocess_dataframe( data )
print "Data cleaned. Ready to go."
#Ok data is pretty clean, what can we do with it?

n, d = data.shape

m = 3*n/4

N = 1000







