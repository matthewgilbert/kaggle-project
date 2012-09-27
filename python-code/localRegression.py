import time

import pandas
import numpy as np
from sklearn.linear_model import LinearRegression

import census_utilities
import geoNN


class LocalRegression( object ):
    """implements a scikitlearn model that finds nearest geographic neighbours and computes a regression. Defaults
    to a global regression if location data is not available.
   
    
    methods:
        fit( data, response, location_data )
        predict( data, location_data)
    
    """
    
    def __init__(self, k = 200, regressor = LinearRegression, params=[] ):
        self.k = k
        self.regressor = regressor
	self.params = params    
    
    def fit(self, data, response, location_data):
        self.data_ = data.values
        self.response_ = response.values
        self.location_data_ = location_data.values
        
        
	self.gnn = geoNN.GeoNNFinder( location_data.values)
        
        return self
        
        
        
    def predict( self, data, location_data):
        
        if location_data.shape[0] != data.shape[0]:
            raise Exception("length of first argument does not equal length of second argument.")
        
        n = location_data.shape[0]
        data = data.values
	reg = self.regressor(*self.params)
        prediction = np.empty( n)
	location = location.values
        for i in range(n):	
	    print i
            location = location_data[i,:] #this can be changed 
            inx = self.gnn.find( location[0], location[1], self.k )
            sub_data = self.data_[inx,:]
            sub_response = self.response_[inx,:]
            reg.fit( sub_data, sub_response )
            prediction[i] = reg.predict( data[i,:] )
	return prediction                
                
