import time

import pandas
import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import Scaler
import census_utilities
import geoNN

from sklearn.feature_selection import SelectPercentile, f_regression


def identity(x):
	return x


class LocalRegression( object ):
    """implements a scikitlearn model that finds nearest geographic neighbours and computes a regression. Defaults
    to a global regression if location data is not available.
   
    
    methods:
        fit( data, response, location_data )
        predict( data, location_data)
    
    """
    
    def __init__(self, k = 200, feature_selection = False, regressor = LinearRegression, verbose=False, params={}, response_f = identity, inv_response_f = identity ):
        self.k = k           
        self.response_f = response_f
        self.inv_response_f = inv_response_f
        self.zero_coef_ = np.zeros( 10000 )
        self.verbose = verbose
	self.feature_selection = feature_selection
        self.selector = SelectPercentile( f_regression, 50 )
        #if regressor is a list of regressor, we need to initialize all of them
        if type( regressor ) == list:
            self.regressor = [ self.regressor[i](**params[i]) for i in range( len(regressor) ) ]
        else:
            self.regressor = [ regressor(**params) ]
        
        
   
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
        
        
        
    def predict( self, data, location_data, weights=None):
        
        if location_data.shape[0] != data.shape[0]:
            raise Exception("length of first argument does not equal length of second argument.")
        
        n = location_data.shape[0]
        try:
            data = data.values
        except:
            pass
        #reg = self.regressor(**self.params)
        prediction = np.zeros( n)
        location_data = location_data.values
	
	weights = weights.values
        argweights_sorted = np.argsort( weights )
        for i in range(n):
            if ( self.verbose and i%100 == 0):
                print i
            #how many estimators should we make, proptional to the percentile of the weight.
            #naive scheme:
            n_estimators = self.trivial_n_estimators( i, argweights_sorted)
            location = location_data[i,:]
            sub_predictions = np.zeros( n_estimators)
            for n_est in range(n_estimators):
                if np.any( pandas.isnull( location ) ):
                    sub_predictions[n_est] =  self.response_.mean()    
                else:
                    inx = self.gnn.find( location[0], location[1], np.floor( (n_est)*np.random.randint(-40,40) + self.k  )  )
                    sub_data = self.data_[inx,:]
                    sub_response = self.response_f( self.response_[inx,:] )
		    to_predict = data[i,:]
                    if self.feature_selection:
                        sub_data = self.selector.fit_transform( self.data_[inx,:], sub_response )
                        to_predict = self.selector.transform( data[i,:] )
                    for reg in self.regressor:    
                        reg.fit(  sub_data , sub_response )
                    try:
                        self.zero_coef_[ np.nonzero( abs(reg.coef_) < .000001 )[0] ] += 1	
                    except:
                        pass
                    #take the average 
                    sub_predictions[n_est] = np.array(  [ reg.predict( to_predict ) for reg in self.regressor] ).mean()
                    
            prediction[i] = sub_predictions.mean()
        #make sure everything is inside [0-100]
        prediction = self.inv_response_f( prediction )
        prediction[ prediction > 100 ] = 99	
        prediction[ prediction < 0 ] = 4
        return prediction                
                
    
    
    def naive_n_estimators(self, i, argweights_sorted):
        """returns the deci-percentile plus 1, i.e. if the weight is the 86th percentile, return 9"""
        n= int( float( argweights_sorted[i] )/len( argweights_sorted) *10 ) + 1
        return n
    
    def trivial_n_estimators(self, i, argweights_sorted):
        return 1
