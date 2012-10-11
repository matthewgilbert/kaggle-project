import time

import pandas
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Scaler
import pp
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
    
    def __init__(self, k = 200, ncpus=1, parallel = False, regressor = LinearRegression, verbose=False, params={}, response_f = identity, inv_response_f = identity ):
        self.k = k           
        self.ncpus = ncpus
        self.parallel = parallel
        self.response_f = response_f
        self.inv_response_f = inv_response_f
        self.zero_coef_ = numpy.zeros( 10000 )
        self.verbose = verbose
        
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
        
    def predict( self, data, location_data):
        if self.parallel:
            #divide the data
            sub_data = numpy.array_split( data, self.ncpus )
            sub_location_data = numpy.array_split( location_data, self.ncpus)

            results = []
            job_server = pp.Server( self.ncpus, ppservers = () )
            for i in range( self.ncpus ):
                f = job_server.submit( self._predict, args =(self, sub_data[i], sub_location[i]), modules=("numpy", "pandas")    )
                results.append( f() )
                
            #concatenate.
            return numpy.concatenate( results )
            
            
        else:
            return self._predict( data, location_data)
            
    def _predict( self, data, location_data):
        if location_data.shape[0] != data.shape[0]:
            raise Exception("length of first argument does not equal length of second argument.")
        
        n = location_data.shape[0]
        try:
            data = data.values
        except:
            pass
        #reg = self.regressor(**self.params)
        prediction = numpy.zero( n)
        location_data = location_data.values
        for i in range(n):    
            location = location_data[i,:] 
            if numpy.any( pandas.isnull( location ) ):
                prediction[i] =  self.response_.mean()    
            else:
                inx = self.gnn.find( location[0], location[1], self.k )
                for reg in self.regressor:    
                    reg.fit(  self.data_[inx,:] , self.response_f( self.response_[inx,:] ) )
                
                try:
                    self.zero_coef_[ numpy.nonzero( abs(reg.coef_) < .000001 )[0] ] += 1	
                except:
                    pass
                #take the average 
                prediction[i] = numpy.array(  [ reg.predict( data[i,:] ) for reg in self.regressor] ).mean()
        #make sure everything is inside [0-100]
        prediction = self.inv_response_f( prediction )
        prediction[ prediction > 100 ] = 99	
        prediction[ prediction < 0 ] = 4
        return prediction                
                

    