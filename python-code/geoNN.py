

import pandas
import numpy as np
from sklearn.linear_models import LinearRegression

import census_utilities

class GeoNN( object ):
    """
    This class implements a geoNN algo that finds the k nearest geographic neighbours. Sometimes that information is not available,
    so we must rely on a different, fall_back, algo. The fallback algo must have .fit and .predict methods.
    
    methods:
     fit(location_data, data, response): fits the data, returns an instance of the model. Ideally all pandas dataframes
     
     
     predict(location_data, data ): returns the predicted response variables, ideally pandas dataframes.

     """
    def __init__(self, kNN, fall_back = LinearRegression() ):
        self.K = kNN
        
        self.fall_back = fall_back
        
        
    def fit(self, data_array, response):
        """
        To inline this with other sklearn models, exdata_array must be a array of the form:
            [ exlocation_data, exdata ]
        
        Ideally both pandas dataframes.
        """
        
        location_data = data_array[0]
        data = data_array[1]
        
        self.data_ = data
        self.location_data_ = location_data
        self.response_ = response
        
        self.fall_back.fit( data, response )
        
        return self
        
    def predict( exdata_array ):
        """
        To inline this with other sklearn models, exdata_array must be a array of the form:
            [ exlocation_data, exdata ]
        
        Ideally both pandas dataframes.
        """
        exlocation_data = exdata_array[0]
        exdata = exdata_array[1]
        
        
        if exlocation_data.shape[0] != exdata.shape[0]:
            raise Exception("length of first argument does not equal length of second argument.")
        
        n = exlocation_data.shape[0]
        
        prediction = np.empty( n )
        
        for i in range(n):
            print i
            if panadas.isnull( exlocation_data.ix[i] ) > 0:
                #missing data, use fall_back
                
                prediction[i] = self.fall_back.predict( exdata.ix[i] )
            else
                l = exlocation_data.ix[i]
                prediction[i] = self.response_.ix[census_utilities.find_geo_NN( l[0], l[1], self.location_data_, self.K ) ].mean()
        
        return prediction
            
        
                
        