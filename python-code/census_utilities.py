"""
census_utlities.py
"""
import numpy as np
import pandas
from sklearn.cross_validation import KFold
import bottleneck as bn


def money2float(s):
   #Converts "$XXX, YYY" to XXXYYY
    try:
        return float( s[1:].replace(",","") )
    except:
       return s


def WMAEscore( prediction, response, weights):
    return abs( weights*(prediction-response) ).sum()/weights.sum()
    
    

def preprocess_dataframe( dataframe ):
    """
    Use this to preprocess crossvalidation data and testing data.
    
    """

    to_remove = ['Mail_Return_Rate_CEN_2010', 'State', 'State_name', 'County_name', 
             'County', 'GIDBG', 'Tract', 'Block_Group', 'Flag', 'weight', 'LATITUDE', 'LONGITUDE']
    for to_rm in to_remove:
        try:
            del dataframe[to_rm]
        except:
            pass

    for col_name in dataframe.columns:


        if "MOE" in col_name:
            del dataframe[col_name] 
            
        
            
        
    to_convert_to_float = ['Med_HHD_Inc_BG_ACS_06_10', 'Med_HHD_Inc_BG_ACSMOE_06_10', 'Med_HHD_Inc_TR_ACS_06_10', 'Med_HHD_Inc_TR_ACSMOE_06_10',
     'Aggregate_HH_INC_ACS_06_10', 'Aggregate_HH_INC_ACSMOE_06_10', 'Med_House_Val_BG_ACS_06_10','Med_House_Val_BG_ACSMOE_06_10','Med_house_val_tr_ACS_06_10',
    'Med_house_val_tr_ACSMOE_06_10', 'Aggr_House_Value_ACS_06_10', 'Aggr_House_Value_ACSMOE_06_10']

    for to_cn in to_convert_to_float:
       try:
         dataframe[to_cn] = dataframe[to_cn].map( money2float )
       except:
	 pass 

     
    for col_name in dataframe.columns:
        #impute here too
        #do a naive filling of missing data
        dataframe[col_name].fillna( value=dataframe[col_name].mean(), inplace=True )

     
    #how much of the data is missing?
    print "Proportion of missing data: %f."%( float( pandas.isnull( dataframe ).sum().sum() )/(dataframe.shape[0]*dataframe.shape[1]) )
        
        

    
    return dataframe
    

    
def cv( model, data, response, weights, K=5, location_data = [], report_training=True):
    """
    model is the scikitlearn model, with parameters already set. This is really to restrictive, eg: if I want to do preprocessing on the 
    cv'ed set, there is no good way.
    """
    kf = KFold( len(response), K , indices = False) 
    print kf

    cv_scores = np.empty( K )
    training_scores = np.empty( K) 
    i = 1
    for train, test in kf:

        training_data = data[ train]
        training_response = response[ train ]
        
        testing_data = data[test]
        testing_response = response[test]

        testing_weights = weights[ test ]
        training_weights = weights[ train ]

	if len(location_data) > 0:
		training_location = [ location_data[train] ]
		testing_location = [ location_data[test] ]

	else:
		training_location = []
		testing_location = []

	fit_args = [ training_data, training_response] + training_location
	predict_args = [ testing_data ] + testing_location	

        model.fit( *fit_args )
        prediction = model.predict( *predict_args )
        cv_scores[i-1] = WMAEscore( prediction, testing_response, testing_weights ) 
	

	print "CV %i: Test accuracy: %0.2f (+/- %0.2f)" % (i, cv_scores[:i].mean(), cv_scores[:i].std() / 2)
	print cv_scores[:i+1]        
	if report_training:

		predict_args = [ training_data] + training_location        
        	training_scores[i-1] = WMAEscore( model.predict( *predict_args  ), training_response, training_weights)
		print "Train acc: %f"%training_scores[i-1]
	print "--------------------------------"
        i += 1 


    print "Test accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() / 2)
    
    if report_training:
        print "Train accuracy: %0.2f (+/- %0.2f)" % (training_scores.mean(), training_scores.std() / 2)

    
        
        
def find_geo_NN( lat, long, location_data, K = 1 ):
    #location_data is a 2-d nx2 numpy array of lat-long coordinates.
    v = (( location_data - np.array( [lat, long] )  )**2).sum(axis=1)
    ix = bn.argpartsort( v, K,axis=None)
    return ix[:K]
    

    

    
