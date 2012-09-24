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

#Ok data is pretty clean, what can we do with it?

n, d = data.shape

#split the data into CV sets. 
K = 5
kf = KFold( len(response), K , indices = False) 
print kf

cv_scores = np.empty( K )
i = 1
for train, test in kf:

    training_data = data[ train]
    training_response = response[ train ]
    
    testing_data = data[ test ]
    testing_response = response[test]

    testing_weights = weights[ test ]
    training_weights = weights[ train ]



    #fit a linear model
    lr = sklm.Lasso(normalize=True,alpha = 1)
    lr.fit( training_data, training_response )
    prediction = lr.predict( testing_data )
    #prediction = 100*np.random.rand( len( testing_response ) ), 33 acc. rate
    
    cv_scores[i-1] = census_utilities.WMAEscore( prediction, testing_response, testing_weights ) 
    print "%d Cross validation: %f" %(i, cv_scores[i-1] )
    i+=1
    
print "Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() / 2)






#open the testing file
#test_data = pandas.read_csv( "testing_filev1.csv")
#test_data = census_utilities.preprocess_dataframe( test_data )


#lr.fit( data, response )
#lr.predict( test_data)










