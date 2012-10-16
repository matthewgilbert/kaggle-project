import numpy as np
import pandas
import localRegression as lclR
import sklearn.linear_model as sklm
from census_utilities import *
from sklearn.svm import SmartSVR

test_file = "test_file_plus_loc_and_response"
try:
	test_data = pandas.load( "../" + test_file + ".pickle" )
except:
	test_data = pandas.read_csv( "../"+ test_file + ".csv" )
	test_data.save( "../"+test_file + ".pickle" )

print "Test data in."

#weights = test_data['weight']
#test_location_data = test_data[ ['LATITUDE', 'LONGITUDE' ] ]

#preprocess data
test_data, test_location_data, test_response, test_weights = preprocess_dataframe( test_data, 0 )#just want the data
del test_data[ test_data.columns[0] ] 
print "Test data cleaned."

print "Shape: ", test_data.shape
#get training_data

training_data = pandas.load( "../TrainingData.pickle" )
print "Training data in."
#training_location_data = training_data[ ['LATITUDE', 'LONGITUDE' ] ]
#training_response = training_data['Mail_Return_Rate_CEN_2010']
#training_weights = training_data['weight']
training_data, training_location_data, training_response, training_weights = preprocess_dataframe( training_data)

print "Shape: ", training_data.shape

### CREATE MODEL ###
#lr = lclR.LocalRegression( k = 1000 )
from sklearn.ensemble import RandomForestRegressor as rfr

#lr = lclR.LocalRegression(k = 1000, response_f = lambda x: np.arcsin(x/100), inv_response_f=lambda x:100*np.sin(x), regressor = SmartSVR, params = {'gamma':0.0001})
lr = lclR.LocalRegression(k = 1000, regressor = sklm.ElasticNet, params = {'alpha':0.001})
lr.fit( training_data, training_response, training_location_data )
print "begin prediction"
prediction = lr.predict( test_data, test_location_data, test_weights )
#remove >100s and <0s
#prediction[ prediction > 100 ] = 99
#prediction[ prediction < 0 ] = 5
np.savetxt( "ElasticNetOct15estimators.csv", prediction, delimiter= "," )
print " COMPLETE." 

