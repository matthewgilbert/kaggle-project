import numpy as np
import pandas
import localRegression as lclR
import sklearn.linear_model as sklm
from census_utilities import *
from sklearn.svm import SmartSVR
from blender import Blender
from sklearn.ensemble import RandomForestRegressor as rfr
import robustEstimator as RE
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
bl = Blender( verbose = True, training_fraction = 0.9)



bl.add_model( lclR.LocalRegression(k = 500 , regressor = SmartSVR, params = {'gamma':0.0001}), "SmrtSVR")
bl.add_model( lclR.LocalRegression(k = 900, regressor = sklm.ElasticNet, params = {'alpha':0.001}), "ElNet500" )
bl.add_model( lclR.LocalRegression(k = 500, regressor = rfr, params={"n_jobs":16, "n_estimators": 24 } ), "rfr50est")
#bl.add_model( lclR.LocalRegression( k=500, regressor = RE.RobustRegression), "robustEst" )
bl.add_model( rfr( n_jobs=150, n_estimators=225 ), "Globalrfr50est")
bl.add_model( lclR.LocalRegression( k=800, regressor = sklm.Ridge, feature_selection = True, params={ 'alpha':0.01, 'normalize':True} ), "RidgeWithFeatureSelection" )

lc = [ training_location_data.values ]

bl.fit( training_data.values, training_response.values, {"SmrtSVR": lc, "ElNet500":lc, "RidgeWithFeatureSelection":lc, "rfr50est":lc, "robustEst":lc } )
 

lc = [test_location_data.values]

print
print "Coefficients:"
print bl.coef_
print "begin prediction"
prediction = bl.predict( test_data.values, { "robustEst":lc, "rfr50est":lc, "SmrtSVR": lc, "ElNet500":lc, "RidgeWithFeatureSelection":lc } )
#remove >100s and <0s
#prediction[ prediction > 100 ] = 99
#prediction[ prediction < 0 ] = 5
np.savetxt( "BlendOct20.csv", prediction, delimiter= "," )
print " COMPLETE." 

