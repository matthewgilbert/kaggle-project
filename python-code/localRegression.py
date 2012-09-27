

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
    
    def __init__(self, k = 200, regressor = LinearRegression ):
        self.k = k
        self.regressor = regressor
        self.global_regressor_ = self.regressor()
        
    def fit(self, data, response, location_data):
        self.data_ = data
        self.response_ = response
        self.location_data_ = location_data
        
        self.global_regressor_.fit( data, response)
        
	self.gnn = geoNN.GeoNNFinder( location_data)
        
        return self
        
        
        
    def predict( self, data, location_data):
        
        if location_data.shape[0] != data.shape[0]:
            raise Exception("length of first argument does not equal length of second argument.")
        
        n = location_data.shape[0]
        
        prediction = np.empty( n)
        for i in range(n):	
	    print i
            location = location_data.values[i,:]
            if np.isnan( location ).any():
                #use global regression
                prediction[i] = self.global_regressor_.predict( data.values[i,:] )
                
            else:
                #gather k near data.
                inx = self.gnn.find( location[0], location[1], self.k )
                sub_data = self.data_.values[inx,:]
                sub_response = self.response_.values[inx,:]
                reg = self.regressor()
                reg.fit( sub_data, sub_response )
                prediction[i] = reg.predict( data.values[i,:] )
	return prediction                
                
