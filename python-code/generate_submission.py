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


#get training_data

training_data = pandas.load( "../TrainingData.pickle" )
print "Training data in."
#training_location_data = training_data[ ['LATITUDE', 'LONGITUDE' ] ]
#training_response = training_data['Mail_Return_Rate_CEN_2010']
#training_weights = training_data['weight']
training_data, training_location_data, training_response, training_weights = preprocess_dataframe( training_data)

### CREATE MODEL ###
#lr = lclR.LocalRegression( k = 1000 )

lr = lclR.LocalRegression(k = 1000, regressor = SmartSVR, verbose = True, params={'cache_size':20000, 'gamma':0.0001, 'verbose':False}, response_f = lambda x: np.arcsin(x/100), inv_response_f=lambda x:100*np.sin(x) )

lr.fit( training_data, training_response, training_location_data )
prediction = lr.predict( test_data, test_location_data )
#remove >100s and <0s
#prediction[ prediction > 100 ] = 99
#prediction[ prediction < 0 ] = 5
np.savetxt( "LatestSubmission.csv", prediction, delimiter= "," )
print file_name, " COMPLETE." 

