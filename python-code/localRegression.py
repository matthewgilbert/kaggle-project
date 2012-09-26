

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
    
    def __init__(self, k = 200, regressor = LinearRegression, params=None ):
        self.k = k
        self.regressor = regressor
        
    def fit(self, data, response, location_data):
        self.data_ = data
        self.response_ = response
        self.location_data_ = location_data
        self.params = params
        
        self.gnn = geoNN.GeoNNFinder( location_data)
        
        return self
        
        
        
    def predict( self, data, location_data):
        
        if location_data.shape[0] != data.shape[0]:
            raise Exception("length of first argument does not equal length of second argument.")
        
        n = location_data.shape[0]
        
        predition = np.empty( n)
        for i in range(n):
            location = location_data.values[i,:]

            #gather k near data.
            inx = self.gnn.find( location[0], location[1], self.k )
            sub_data = data.values[inx,:]
            sub_response = response.values[inx,:]
            reg = self.regressor(*self.params)
            reg.fit( sub_data, sub_response )
            prediction[i] = reg.predict( data.values[i,:] )
                
                
