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


training_file = "training_file_plus_location.csv"
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
location_data = data[ ['LATITUDE','LONGITUDE'] ]
#location_data = pandas.read_csv( "../training_locations.csv")

#del data['Mail_Return_Rate_CEN_2010']
#del data['weight']
#del data[ data.columns[0] ]
#preprocess data.
data, location_data = preprocess_dataframe( data, location_data )
print "Data cleaned. TODO: check cleaning algos"
#Ok data is pretty clean, what can we do with it?

n, d = data.shape

m = 3*n/4

N = 1000







