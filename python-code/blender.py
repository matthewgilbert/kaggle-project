
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from time import clock


class Blender( object):
    """
        This class implements a linear blend of different models.
    
    
        methods:
            fit( data, response, dict_of_additional_variables )
            add_model( model, name)
            predict( new_data, dict_of_additional_variables )
        
        
        attributes:
            coefs_
            
    """
    
    
    def __init__( self, blender = LinearRegression(), training_fraction = 0.8, verbose = False):
            self.blender = blender
            self.training_fraction = training_fraction
            self.verbose = verbose
            self.models = dict()
            
    
    
    def add_model(self, model, name=None):
        """ 
            model: a sklearn model that exposes the methods fit & predict. 
            name: a name to specify the model, eg "ElNet500alpha"
        
        """
        if not name:
            name = "%d"%(len(self.models) +1 )
        self.models[name] = model
        return
        
    def del_model(self, name ):
	try:	
	   del self.model[name]
	except KeyError:
	   print "Model %s not in blender."%name
	return
	
 
    def fit(self, data, response, dict_of_additional_variables=None):
        """
            data: the data matix, shape (n,d)
            response: the response vector (n,)
            dict_of_additional_variables:
                a dictionary with the keys the model names (optional to include), and the items are of the form:
                        {"train":[ items to be included in training], "test":[items to be included in testing] }
        """
        
        #split the data to held-in and held-out. 
        training_data, blend_data, training_response, blend_response = train_test_split( data, response, test_size = 1- self.training_fraction)
        X = np.zeros( (blend_response.shape[0], len( self.models ) ) )

        if self.verbose:
            print "Shape of training data vs blending data: ", training_data.shape, blend_data.shape 
        #train the models.
        i = 0
        for name, model in sorted( self.models.iteritems() ):
            start = clock()
            try:
                model.fit( training_data, training_response, *dict_of_additional_variables[name]["train"] )
                X[:, i] = model.predict( blend_data, *dict_of_additional_variables[name]["test"] )
                
            except (KeyError, TypeError):
                model.fit( training_data, training_response)
                X[:, i] = model.predict( blend_data )
            i+=1
            if self.verbose:
                print "Finished model %s. Took %.3f seconds."%(name, clock() - start)

        #create a data matrix X of predictions.
        if self.verbose:
            print "Fitting finished, starting blending."
        
        self.blender.fit( X, blend_response )
        self.coef_ = self.blender.coef_
        
        self._fit_training_data = training_data
        self._fit_blend_data = blend_data
        self._fit_training_response = training_response
        self._fit_blend_response = blend_response
        
        if self.verbose:
            print "Done fitting"
            
        return self
            
    def predict( self, data, dict_of_additional_variables=None):
        
        X = np.zeros( (data.shape[0], len( self.models ) ) )
        i = 0
        for name, model in sorted( self.models.iteritems() ):
            X[:,i] = model.predict( data )
            if self.verbose:
                print "Finished model %s."%(name)
	    i+=1	
        return self.blender.predict( X )
        
        
