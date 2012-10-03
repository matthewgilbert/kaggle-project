import time

import pandas
import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import Scaler
import census_utilities
import geoNN



def identity(x):
	return x


class LocalRegression( object ):
    """implements a scikitlearn model that finds nearest geographic neighbours and computes a regression. Defaults
    to a global regression if location data is not available.
   
    
    methods:
        fit( data, response, location_data )
        predict( data, location_data)
    
    """
    
    def __init__(self, k = 200, regressor = LinearRegression, params={}, scale=True, response_f = identity, inv_response_f = identity ):
        self.k = k
        self.regressor = regressor
	self.params = params    
	if scale:
		self.Scaler = Scaler()
	
	self.response_f = response_f
	self.inv_response_f = inv_response_f
	self.zero_coef_ = np.zeros( 10000 )
   
    def __str__(self):
	return "%dK%sLocalRegression"%(self.k, self.regressor.__name__.replace(" ", "" ))
 
    def fit(self, data, response, location_data):
        try:
		#incase I pass in a numpy array or pandas df
		self.data_ = data.values
        except:
		self.data_ = data
	self.response_ = response.values
        self.location_data_ = location_data.values
        
        
	self.gnn = geoNN.GeoNNFinder( location_data.values)
        
        return self
        
        
        
    def predict( self, data, location_data):
        
        if location_data.shape[0] != data.shape[0]:
            raise Exception("length of first argument does not equal length of second argument.")
        
        n = location_data.shape[0]
        try:
		data = data.values
	except:
		pass
	reg = self.regressor(**self.params)
        prediction = np.empty( n)
	location_data = location_data.values
        for i in range(n):
	    if i%100 == 0:
	    	print i	
            location = location_data[i,:] 
      	    if np.any( pandas.isnull( location ) ):
		prediction[i] =  self.response_.mean()    

	    else:
	    	inx = self.gnn.find( location[0], location[1], self.k )
          	reg.fit(  self.data_[inx,:] , self.response_f( self.response_[inx,:] ) )
		try:
		    self.zero_coef_[ np.nonzero( abs(reg.coef_) < .000001 )[0] ] += 1	
		except:
		    pass
            	prediction[i] = reg.predict( data[i,:]  )
	#make sure everything is inside [0-100]
	prediction = self.inv_response_f( prediction )
	prediction[ prediction > 100 ] = 99	
	prediction[ prediction < 0 ] = 4
	return prediction                
                
