import numpy as np
import pandas
import localRegression as lclR
import sklearn.linear_model as sklm
from census_utilities import *

test_file = "test_file_plus_location"
try:
	test_data = pandas.load( "../" + test_file + ".pickle" )
except:
	test_data = pandas.read_csv( "../"+ test_file + ".csv" )
	test_data.save( "../"+test_file + ".pickle" )

print "Test data in."

weights = test_data['weight']
test_location_data = test_data[ ['LATITUDE', 'LONGITUDE' ] ]

#preprocess data
test_data = preprocess_dataframe( test_data )
print "Test data cleaned."


#get training_data

training_data = pandas.load( "../training_file_plus_location.pickle" )
training_location_data = training_data[ ['LATITUDE', 'LONGITUDE' ] ]
training_response = training_data['Mail_Return_Rate_CEN_2010']
training_weights = training_data['weight']
training_data = preprocess_dataframe( training_data)

### CREATE MODEL ###
elnet = sklm.ElasticNet
lr = lclR.LocalRegression( k = 1000, regressor = elnet, params = {'alpha':0.0001, 'rho':0.9 } )
lr.fit( training_data, training_response, training_location_data )
prediction = lr.predict( test_data, test_location_data )
#remove >100s and <0s
prediction[ prediction > 100 ] = 99
prediction[ prediction < 0 ] = 5
file_name = "Submssion" + lr.__str__ + ".csv"
np.savetxt( file_name, prediction, delimiter= "," )
print file_name, " COMPLETE." 

