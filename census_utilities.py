"""
census_utlities.py
"""



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
             'County', 'GIDBG', 'Tract', 'Block_Group', 'Flag', 'weight']
    for to_rm in to_remove:
        try:
            del dataframe[to_rm]
        except:
            pass

        
    to_convert_to_float = ['Med_HHD_Inc_BG_ACS_06_10', 'Med_HHD_Inc_BG_ACSMOE_06_10', 'Med_HHD_Inc_TR_ACS_06_10', 'Med_HHD_Inc_TR_ACSMOE_06_10',
     'Aggregate_HH_INC_ACS_06_10', 'Aggregate_HH_INC_ACSMOE_06_10', 'Med_House_Val_BG_ACS_06_10','Med_House_Val_BG_ACSMOE_06_10','Med_house_val_tr_ACS_06_10',
    'Med_house_val_tr_ACSMOE_06_10', 'Aggr_House_Value_ACS_06_10', 'Aggr_House_Value_ACSMOE_06_10']

    for to_cn in to_convert_to_float:
        dataframe[to_cn] = dataframe[to_cn].map( money2float )
        

    #do a naive filling of missing data
    dataframe.fillna( method="bfill", inplace=True)
    dataframe.fillna( method="ffill", inplace=True)
    
    return dataframe