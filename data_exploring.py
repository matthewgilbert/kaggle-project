"""
Kaggle Census competition
"""

import math

import pandas 
import sklearn.linear_model as sklm

import census_utilities 


#open training_file
data = pandas.read_csv( "training_filev1.csv")
response = data['Mail_Return_Rate_CEN_2010']
weights = data['weight']





#preprocess data.
data = census_utilities.preprocess_dataframe( data )

#Ok data is pretty clean, what can we do with it?

n, d = data.shape

#split the data into two. Replace this with cross-validation eventually.
training_data = data[ :math.floor( 0.8*n) ]
training_response = response[ :math.floor( 0.8*n) ]

test_data = data[ math.floor( 0.8*n):]
test_response = response[ math.floor( 0.8*n): ]

training_weights = weights[:math.floor( 0.8*n) ]
testing_weights = weights[ math.floor( 0.8*n): ]



#fit a linear model
print "Create a linear model."
lr = sklm.LinearRegression( normalize = True )
lr.fit( training_data, training_response )

prediction = lr.predict( test_data )

print "Cross validation: %f" %census_utilities.WMAEscore( prediction, test_response, testing_weights )


#open the testing file
test_data = pandas.read_csv( "testing_filev1.csv")
test_data = census_utilities.preprocess_dataframe( test_data )


lr.fit( data, response )
lr.predict( test_data)










