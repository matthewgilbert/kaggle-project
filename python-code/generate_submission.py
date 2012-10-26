import numpy as np
import os
import pandas
import localRegression as lclR
import sklearn.linear_model as sklm
from census_utilities import *
from sklearn.svm import SmartSVR
from blender import Blender
from sklearn.ensemble import RandomForestRegressor as rfr
import robustEstimator as RE
from datetime import datetime

test_file = "test_file_plus_loc_and_response"
try:
	test_data = pandas.load( "../" + test_file + ".pickle" )
except:
	test_data = pandas.read_csv( "../"+ test_file + ".csv" )
	test_data.save( "../"+test_file + ".pickle" )

print datetime.now()

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
bl = Blender( verbose = True, training_fraction = 0.95)



bl.add_model( lclR.LocalRegression(k = 500 , regressor = SmartSVR, params = {'gamma':0.0001}), "SmrtSVR")
bl.add_model( lclR.LocalRegression(k = 500 , regressor = SmartSVR, params = {'gamma':0.001, "C":50}), "LooseSmrtSVR")
bl.add_model( lclR.LocalRegression(k = 500, regressor = sklm.ElasticNet, params = {'alpha':0.0001, "normalize":True}), "ElNet500" )
bl.add_model( lclR.LocalRegression(k = 500, regressor = rfr, params={"n_jobs":5, "n_estimators": 10 } ), "rfr50est")
bl.add_model( lclR.LocalRegression( k=500, regressor = sklm.Lasso, params={"alpha":0.01, "normalize":True}), "lasso" )
bl.add_model( rfr( n_jobs=150, n_estimators=200 ), "Globalrfr50est")
bl.add_model( lclR.LocalRegression( k=750, regressor = sklm.Ridge, feature_selection = True, params={ 'alpha':0.001, 'normalize':True} ), "RidgeWithFeatureSelection" )
bl.add_model( lclR.LocalRegression( k=750, regressor = sklm.Ridge, feature_selection = False, params={"alpha":0.1, "normalize":True}), "Ridge")
bl.add_model( lclR.LocalRegression( k = 250, regressor = sklm.Ridge, feature_selection = False, params={"alpha":1.5, "normalize":True} ), "LocalRidge")
lc = [ training_location_data.values ]

bl.fit( training_data.values, training_response.values, {"SmrtSVR": lc, "ElNet500":lc, "RidgeWithFeatureSelection":lc, "rfr50est":lc, "lasso":lc, "Ridge":lc, "LooseSmrtSVR":lc, "LocalRidge":lc } )
 

lc = [test_location_data.values]

print
print "Coefficients:"
print bl.coef_
print "begin prediction"
prediction = bl.predict( test_data.values, { "lasso":lc, "rfr50est":lc, "SmrtSVR": lc, "ElNet500":lc, "RidgeWithFeatureSelection":lc, "Ridge":lc, "LooseSmrtSVR":lc, "LocalRidge":lc } )
#remove >100s and <0s
#prediction[ prediction > 100 ] = 99
#prediction[ prediction < 0 ] = 5
file_name = "BlendOct25NonLinear.csv"
np.savetxt(file_name, prediction, delimiter= "," )
print " COMPLETE." 
#mail it to me
cmd_line = """echo "%s is complete." | mail -s "High Fructose" cam.davidson.pilon@gmail.com"""%file_name
os.system(cmd_line)



