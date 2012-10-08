"""
census_utlities.py
"""
import numpy as np
import pandas
from sklearn.cross_validation import KFold
import bottleneck as bn
import scipy.stats as stats


def money2float(s):
   #Converts "$XXX, YYY" to XXXYYY. Fuck it, lets log it too.
    try:
        return np.log( float( s[1:].replace(",","") )) 
    except:
       return s


def WMAEscore( prediction, response, weights):
    return abs( weights*(prediction-response) ).sum()/weights.sum()
    
 

 
def generate_features( dataframe ):

    #population density /sq mile
    #dataframe['Population Density'] = dataframe['Tot_Population_CEN_2010']/( dataframe['LAND_AREA'] ).astype( np.float64)
    
    #housing density
    #dataframe['Housing Density'] = dataframe['Tot_Occp_Units_CEN_2010']/( dataframe['LAND_AREA'] ).astype( np.float64)
    
    #bilingual/spanish 
    
    #dataframe['Diff_Of_Spanish_ACS_Prop_to_Spanish_Ballot_Prop'] = dataframe['ENG_VW_SPAN_ACS_06_10']/(dataframe['Tot_Occp_Units_ACS_06_10'] ).astype( np.float64)  - dataframe['BILQ_Mailout_count_CEN_2010']/(dataframe['Tot_Occp_Units_CEN_2010'] ).astype( np.float64)

    #dataframe['Diff_Of_Spanish_CEN_Prop_to_Spanish_Ballot_Prop'] = dataframe['Hispanic_CEN_2010']/(dataframe['Tot_Population_CEN_2010'] ).astype( np.float64) - dataframe['BILQ_Mailout_count_CEN_2010']/(dataframe['Tot_Occp_Units_CEN_2010'] ).astype( np.float64)
    
    #these were shown to be interesting in lasso 
    dataframe["URBANIZED_AREA_*MLT_U2_9_STRC_A"] = dataframe["URBANIZED_AREA_POP_CEN_2010/Tot_Population_CEN_2010"]*dataframe["MLT_U2_9_STRC_ACS_06_10/Tot_Occp_Units_ACS_06_10"]
    dataframe["RURAL_POP_CEN_2*Hispanic_CEN_20"] = dataframe["RURAL_POP_CEN_2010/Tot_Population_CEN_2010"]*dataframe["Hispanic_CEN_2010/Tot_Population_CEN_2010"]
    dataframe["RURAL_POP_CEN_2*Prs_Blw_Pov_Lev"] = dataframe["RURAL_POP_CEN_2010/Tot_Population_CEN_2010"]*dataframe["Prs_Blw_Pov_Lev_ACS_06_10/Tot_Population_ACS_06_10"]
    dataframe["Females_CEN_201*2000_response"] = dataframe["Females_CEN_2010/Tot_Population_CEN_2010"]*dataframe["2000_response"]
    dataframe["Pop_under_5_CEN*Pop_18_24_CEN_2"] = dataframe["Pop_under_5_CEN_2010/Tot_Population_CEN_2010"]*dataframe["Pop_18_24_CEN_2010/Tot_Population_CEN_2010"]
    dataframe["Pop_18_24_CEN_2*NH_Blk_alone_CE"] = dataframe["Pop_18_24_CEN_2010/Tot_Population_CEN_2010"]*dataframe["NH_Blk_alone_CEN_2010/Tot_Population_CEN_2010"]
    dataframe["Pop_18_24_CEN_2*NH_SOR_alone_CE"] = dataframe["Pop_18_24_CEN_2010/Tot_Population_CEN_2010"]*dataframe["NH_SOR_alone_CEN_2010/Tot_Population_CEN_2010"]
    dataframe["Pop_18_24_CEN_2*Pov_Univ_ACS_06"] = dataframe["Pop_18_24_CEN_2010/Tot_Population_CEN_2010"]*dataframe["Pov_Univ_ACS_06_10/Tot_Population_ACS_06_10"]
    dataframe["Pop_18_24_CEN_2*Tot_Prns_in_HHD"] = dataframe["Pop_18_24_CEN_2010/Tot_Population_CEN_2010"]*dataframe["Tot_Prns_in_HHD_CEN_2010/Tot_Occp_Units_CEN_2010"]
    dataframe["Pop_18_24_CEN_2*Renter_Occp_HU_"] = dataframe["Pop_18_24_CEN_2010/Tot_Population_CEN_2010"]*dataframe["Renter_Occp_HU_CEN_2010/Tot_Occp_Units_CEN_2010"]
    dataframe["Pop_18_24_CEN_2*MLT_U2_9_STRC_A"] = dataframe["Pop_18_24_CEN_2010/Tot_Population_CEN_2010"]*dataframe["MLT_U2_9_STRC_ACS_06_10/Tot_Occp_Units_ACS_06_10"]
    dataframe["Pop_25_44_CEN_2*Female_No_HB_CE"] = dataframe["Pop_25_44_CEN_2010/Tot_Population_CEN_2010"]*dataframe["Female_No_HB_CEN_2010/Tot_Occp_Units_CEN_2010"]
    dataframe["Pop_65plus_CEN_*2000_response"] = dataframe["Pop_65plus_CEN_2010/Tot_Population_CEN_2010"]*dataframe["2000_response"]
    dataframe["Hispanic_CEN_20*NH_White_alone_"] = dataframe["Hispanic_CEN_2010/Tot_Population_CEN_2010"]*dataframe["NH_White_alone_CEN_2010/Tot_Population_CEN_2010"]
    dataframe["Hispanic_CEN_20*Tot_Vacant_Unit"] = dataframe["Hispanic_CEN_2010/Tot_Population_CEN_2010"]*dataframe["Tot_Vacant_Units_ACS_06_10/Tot_Occp_Units_ACS_06_10"]
    dataframe["NH_White_alone_*NH_AIAN_alone_C"] = dataframe["NH_White_alone_CEN_2010/Tot_Population_CEN_2010"]*dataframe["NH_AIAN_alone_CEN_2010/Tot_Population_CEN_2010"]
    dataframe["NH_White_alone_*Othr_Lang_ACS_0"] = dataframe["NH_White_alone_CEN_2010/Tot_Population_CEN_2010"]*dataframe["Othr_Lang_ACS_06_10/Tot_Population_ACS_06_10"]
    dataframe["NH_White_alone_*College_ACS_06_"] = dataframe["NH_White_alone_CEN_2010/Tot_Population_CEN_2010"]*dataframe["College_ACS_06_10/Pop_25yrs_Over_ACS_06_10"]
    dataframe["NH_White_alone_*MrdCple_Fmly_HH"] = dataframe["NH_White_alone_CEN_2010/Tot_Population_CEN_2010"]*dataframe["MrdCple_Fmly_HHD_CEN_2010/Tot_Occp_Units_CEN_2010"]
    dataframe["NH_White_alone_*Owner_Occp_HU_C"] = dataframe["NH_White_alone_CEN_2010/Tot_Population_CEN_2010"]*dataframe["Owner_Occp_HU_CEN_2010/Tot_Occp_Units_CEN_2010"]
    dataframe["NH_Blk_alone_CE*NH_SOR_alone_CE"] = dataframe["NH_Blk_alone_CEN_2010/Tot_Population_CEN_2010"]*dataframe["NH_SOR_alone_CEN_2010/Tot_Population_CEN_2010"]
    dataframe["NH_Blk_alone_CE*MLT_U2_9_STRC_A"] = dataframe["NH_Blk_alone_CEN_2010/Tot_Population_CEN_2010"]*dataframe["MLT_U2_9_STRC_ACS_06_10/Tot_Occp_Units_ACS_06_10"]
    dataframe["NH_AIAN_alone_C*Med_HHD_Inc_TR_"] = dataframe["NH_AIAN_alone_CEN_2010/Tot_Population_CEN_2010"]*dataframe["Med_HHD_Inc_TR_ACS_06_10"]
    dataframe["NH_SOR_alone_CE*Tot_Prns_in_HHD"] = dataframe["NH_SOR_alone_CEN_2010/Tot_Population_CEN_2010"]*dataframe["Tot_Prns_in_HHD_CEN_2010/Tot_Occp_Units_CEN_2010"]
    dataframe["Othr_Lang_ACS_0*MLT_U2_9_STRC_A"] = dataframe["Othr_Lang_ACS_06_10/Tot_Population_ACS_06_10"]*dataframe["MLT_U2_9_STRC_ACS_06_10/Tot_Occp_Units_ACS_06_10"]
    dataframe["College_ACS_06_*Owner_Occp_HU_C"] = dataframe["College_ACS_06_10/Pop_25yrs_Over_ACS_06_10"]*dataframe["Owner_Occp_HU_CEN_2010/Tot_Occp_Units_CEN_2010"]
    dataframe["Prs_Blw_Pov_Lev*Mobile_Homes_AC"] = dataframe["Prs_Blw_Pov_Lev_ACS_06_10/Tot_Population_ACS_06_10"]*dataframe["Mobile_Homes_ACS_06_10/Tot_Occp_Units_ACS_06_10"]
    dataframe["ENG_VW_INDO_EUR*HHD_PPL_Und_18_"] = dataframe["ENG_VW_INDO_EURO_ACS_06_10/Tot_Occp_Units_ACS_06_10"]*dataframe["HHD_PPL_Und_18_CEN_2010/Tot_Occp_Units_CEN_2010"]
    dataframe["ENG_VW_INDO_EUR*Rel_Child_Under"] = dataframe["ENG_VW_INDO_EURO_ACS_06_10/Tot_Occp_Units_ACS_06_10"]*dataframe["Rel_Child_Under_6_CEN_2010/Tot_Occp_Units_CEN_2010"]
    dataframe["ENG_VW_INDO_EUR*MLT_U2_9_STRC_A"] = dataframe["ENG_VW_INDO_EURO_ACS_06_10/Tot_Occp_Units_ACS_06_10"]*dataframe["MLT_U2_9_STRC_ACS_06_10/Tot_Occp_Units_ACS_06_10"]
    dataframe["MrdCple_Fmly_HH*Sngl_Prns_HHD_C"] = dataframe["MrdCple_Fmly_HHD_CEN_2010/Tot_Occp_Units_CEN_2010"]*dataframe["Sngl_Prns_HHD_CEN_2010/Tot_Occp_Units_CEN_2010"]
    dataframe["MrdCple_Fmly_HH*2000_response"] = dataframe["MrdCple_Fmly_HHD_CEN_2010/Tot_Occp_Units_CEN_2010"]*dataframe["2000_response"]
    dataframe["Female_No_HB_CE*MLT_U2_9_STRC_A"] = dataframe["Female_No_HB_CEN_2010/Tot_Occp_Units_CEN_2010"]*dataframe["MLT_U2_9_STRC_ACS_06_10/Tot_Occp_Units_ACS_06_10"]
    dataframe["NonFamily_HHD_C*HHD_PPL_Und_18_"] = dataframe["NonFamily_HHD_CEN_2010/Tot_Occp_Units_CEN_2010"]*dataframe["HHD_PPL_Und_18_CEN_2010/Tot_Occp_Units_CEN_2010"]
    dataframe["HHD_PPL_Und_18_*Mobile_Homes_AC"] = dataframe["HHD_PPL_Und_18_CEN_2010/Tot_Occp_Units_CEN_2010"]*dataframe["Mobile_Homes_ACS_06_10/Tot_Occp_Units_ACS_06_10"]
    dataframe["Tot_Prns_in_HHD*Renter_Occp_HU_"] = dataframe["Tot_Prns_in_HHD_CEN_2010/Tot_Occp_Units_CEN_2010"]*dataframe["Renter_Occp_HU_CEN_2010/Tot_Occp_Units_CEN_2010"]
    dataframe["Tot_Prns_in_HHD*MLT_U2_9_STRC_A"] = dataframe["Tot_Prns_in_HHD_CEN_2010/Tot_Occp_Units_CEN_2010"]*dataframe["MLT_U2_9_STRC_ACS_06_10/Tot_Occp_Units_ACS_06_10"]
    dataframe["Rel_Child_Under*Tot_Housing_Uni"] = dataframe["Rel_Child_Under_6_CEN_2010/Tot_Occp_Units_CEN_2010"]*dataframe["Tot_Housing_Units_CEN_2010/Tot_Occp_Units_CEN_2010"]
    dataframe["Rel_Child_Under*Mobile_Homes_AC"] = dataframe["Rel_Child_Under_6_CEN_2010/Tot_Occp_Units_CEN_2010"]*dataframe["Mobile_Homes_ACS_06_10/Tot_Occp_Units_ACS_06_10"]
    dataframe["HHD_Moved_in_AC*Renter_Occp_HU_"] = dataframe["HHD_Moved_in_ACS_06_10/Tot_Occp_Units_ACS_06_10"]*dataframe["Renter_Occp_HU_CEN_2010/Tot_Occp_Units_CEN_2010"]
    dataframe["Owner_Occp_HU_C*2000_response"] = dataframe["Owner_Occp_HU_CEN_2010/Tot_Occp_Units_CEN_2010"]*dataframe["2000_response"]

    return dataframe
 
 

def preprocess_dataframe( dataframe, training=1 ):
    """
    Use this to preprocess crossvalidation data and testing data.
    
    """
    
    #some data points are giving us problems, lets delete them. 
    if training:
        print "dropping some data points."
        print "Previous size: %d"%dataframe.shape[0]
        ix = np.nonzero( dataframe['Tot_Population_ACS_06_10'] == 0)[0]
        dataframe = dataframe.drop( ix )

    	#dataframe = dataframe.dropna(subset=['2000_response'])
    	print "New Size: %d"%dataframe.shape[0]
    	del dataframe[ dataframe.columns[0] ]    
    
    	del dataframe[ dataframe.columns[0] ]    
    
    weights = None
    response = None
    location_data = dataframe[ ['LATITUDE', 'LONGITUDE'] ] 
    if training:
    	response = dataframe['Mail_Return_Rate_CEN_2010']
    	weights = dataframe['weight']
 
    
    to_remove = ['TEA_Mail_Out_Mail_Back_CEN_2010','TEA_Update_Leave_CEN_2010', 'BILQ_Mailout_count_CEN_2010', 'Mail_Return_Rate_CEN_2010', 'State', 'State_name', 'County_name', 
             'County', 'LAND_AREA', 'AIAN_LAND', 'GIDBG', 'Tract', 'Block_Group', 'Flag', 'weight', 'LATITUDE', 'LONGITUDE', 'MailBack_Area_Count_CEN_2010']
             
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
    
    
    return transform_data( dataframe ), location_data, response, weights

    
def transform_data( dataframe ):
    """
    This creates proportions out of the data.
    """
    
    #education
    for category in ['Not_HS_Grad_ACS_06_10',  'College_ACS_06_10'] :
            dataframe[category] = dataframe[category]/dataframe['Pop_25yrs_Over_ACS_06_10'].astype(np.float64)
            dataframe.rename( columns = { category: category + "/" + "Pop_25yrs_Over_ACS_06_10" }, inplace = True )
    
    del dataframe["Males_ACS_06_10"]
    del dataframe["Females_ACS_06_10"]
    del dataframe["Pop_under_5_ACS_06_10"]
    del dataframe["Pop_5_17_ACS_06_10"]
    del dataframe["Pop_18_24_ACS_06_10"]
    del dataframe["Pop_25_44_ACS_06_10"]
    del dataframe["Pop_45_64_ACS_06_10"]
    del dataframe["Pop_65plus_ACS_06_10"]
    del dataframe["Hispanic_ACS_06_10"]
    del dataframe["NH_White_alone_ACS_06_10"]
    del dataframe["NH_Blk_alone_ACS_06_10"]
    del dataframe["NH_AIAN_alone_ACS_06_10"]
    del dataframe["NH_Asian_alone_ACS_06_10"]
    del dataframe["NH_NHOPI_alone_ACS_06_10"]
    del dataframe["NH_SOR_alone_ACS_06_10"]
    del dataframe["Pop_5yrs_Over_ACS_06_10"]
    acs_population = dataframe['Tot_Population_ACS_06_10'].astype(np.float64)
    acs_populations_to_normalize = [
        #    "Males_ACS_06_10",
            #"Females_ACS_06_10", 
            #"Pop_under_5_ACS_06_10",
            #"Pop_5_17_ACS_06_10",
            #"Pop_18_24_ACS_06_10",
            #"Pop_25_44_ACS_06_10",
            #"Pop_45_64_ACS_06_10",
             # "Pop_65plus_ACS_06_10",
              #"Hispanic_ACS_06_10",
              #"NH_White_alone_ACS_06_10",
              #"NH_Blk_alone_ACS_06_10",
                #"NH_AIAN_alone_ACS_06_10",
                #"NH_Asian_alone_ACS_06_10",
            #"NH_NHOPI_alone_ACS_06_10",
            #"NH_SOR_alone_ACS_06_10",
            #"Pop_5yrs_Over_ACS_06_10",
              "Othr_Lang_ACS_06_10",
              "Pop_25yrs_Over_ACS_06_10",
              "Prs_Blw_Pov_Lev_ACS_06_10",
              "Pov_Univ_ACS_06_10",
              "Pop_1yr_Over_ACS_06_10",
              "Diff_HU_1yr_Ago_ACS_06_10",
             ]
    
    for category in acs_populations_to_normalize:
        dataframe[category] = dataframe[category]/acs_population
        dataframe.rename( columns = { category: category + "/" + 'Tot_Population_ACS_06_10' }, inplace = True )        
    del dataframe['Tot_Population_ACS_06_10']  
              
              
    del dataframe["Not_MrdCple_HHD_ACS_06_10"]   
    del dataframe["Tot_Prns_in_HHD_ACS_06_10"]   
    del dataframe["Owner_Occp_HU_ACS_06_10"]   
    del dataframe["Renter_Occp_HU_ACS_06_10"]   
    del dataframe["NonFamily_HHD_ACS_06_10"]   
    del dataframe["Sngl_Prns_HHD_ACS_06_10"]   
    del dataframe["MrdCple_Fmly_HHD_ACS_06_10"]   
    del dataframe["Rel_Child_Under_6_ACS_06_10"]   
    del dataframe["Female_No_HB_ACS_06_10"]
    del dataframe["Tot_Housing_Units_ACS_06_10"] 
    del dataframe["HHD_PPL_Und_18_ACS_06_10"]  
    #denominator is Tot_Occp_Units_ACS_06_10
    acs_households = dataframe["Tot_Occp_Units_ACS_06_10"].astype(np.float64)          
    acs_households_to_normalize = [
                "ENG_VW_SPAN_ACS_06_10",
                "ENG_VW_INDO_EURO_ACS_06_10",
                "ENG_VW_API_ACS_06_10",
                "ENG_VW_OTHER_ACS_06_10",
                "ENG_VW_ACS_06_10",
                "Rel_Family_HHD_ACS_06_10",
                #"MrdCple_Fmly_HHD_ACS_06_10",
                #"Not_MrdCple_HHD_ACS_06_10", 
                #"Female_No_HB_ACS_06_10",
                #"NonFamily_HHD_ACS_06_10",
                #"Sngl_Prns_HHD_ACS_06_10",
                #"HHD_PPL_Und_18_ACS_06_10", 
                #"Tot_Prns_in_HHD_ACS_06_10",
                #"Rel_Child_Under_6_ACS_06_10",
                "HHD_Moved_in_ACS_06_10", 
                "PUB_ASST_INC_ACS_06_10", 
                "Aggregate_HH_INC_ACS_06_10",
                #"Tot_Housing_Units_ACS_06_10",
                #"Renter_Occp_HU_ACS_06_10", 
                #"Owner_Occp_HU_ACS_06_10",
                "Single_Unit_ACS_06_10",
                "MLT_U2_9_STRC_ACS_06_10",
                "MLT_U10p_ACS_06_10",
                "Mobile_Homes_ACS_06_10",
                "Crowd_Occp_U_ACS_06_10", 
                "Occp_U_NO_PH_SRVC_ACS_06_10",
                "No_Plumb_ACS_06_10", 
                "Built_Last_5_yrs_ACS_06_10",
                "Tot_Vacant_Units_ACS_06_10",
                "Aggr_House_Value_ACS_06_10",
                ]
    for category in acs_households_to_normalize:
        dataframe[category] = dataframe[category]/acs_households
        dataframe.rename( columns = { category: category + "/" + "Tot_Occp_Units_ACS_06_10" }, inplace = True )

    del dataframe['Tot_Occp_Units_ACS_06_10']
    
    

    
    
    #we should reduce redundancy
    del dataframe['Males_CEN_2010']
    del dataframe['Non_Inst_GQ_CEN_2010']
    
    #census
    census_population = dataframe['Tot_Population_CEN_2010'].astype(np.float64)
    census_pop_to_normalize = [
        'URBANIZED_AREA_POP_CEN_2010',
        'URBAN_CLUSTER_POP_CEN_2010',
        'RURAL_POP_CEN_2010',
        'Females_CEN_2010',
        'Pop_under_5_CEN_2010', 
        'Pop_5_17_CEN_2010',
        'Pop_18_24_CEN_2010',
        'Pop_25_44_CEN_2010',
        'Pop_45_64_CEN_2010',
        'Pop_65plus_CEN_2010',
        'Tot_GQ_CEN_2010',
        'Inst_GQ_CEN_2010',
        #'Non_Inst_GQ_CEN_2010', 
        'Hispanic_CEN_2010',
        'NH_White_alone_CEN_2010', 
        'NH_Blk_alone_CEN_2010',
        'NH_AIAN_alone_CEN_2010',
        'NH_Asian_alone_CEN_2010', 
        'NH_NHOPI_alone_CEN_2010',
        'NH_SOR_alone_CEN_2010',]
        
    for category in census_pop_to_normalize:
        dataframe[category] = dataframe[category]/census_population
        dataframe.rename( columns = { category: category + "/" + 'Tot_Population_CEN_2010' }, inplace = True )

    del dataframe['Tot_Population_CEN_2010']     
    


    del dataframe['Not_MrdCple_HHD_CEN_2010']
    census_households = dataframe['Tot_Occp_Units_CEN_2010'].astype(np.float64)
    census_hhd_to_normalize = [
        'Rel_Family_HHDS_CEN_2010',
        "MrdCple_Fmly_HHD_CEN_2010", 
        #"Not_MrdCple_HHD_CEN_2010",
        'Female_No_HB_CEN_2010',
        'NonFamily_HHD_CEN_2010',
        'Sngl_Prns_HHD_CEN_2010',
        'HHD_PPL_Und_18_CEN_2010',
        'Tot_Prns_in_HHD_CEN_2010',
        'Tot_Vacant_Units_CEN_2010',
        'Owner_Occp_HU_CEN_2010',
        'Rel_Child_Under_6_CEN_2010',
        'Renter_Occp_HU_CEN_2010',
        'Tot_Housing_Units_CEN_2010',
        ]
    
    for category in census_hhd_to_normalize:
        dataframe[category] = dataframe[category] / census_households
        dataframe.rename( columns = { category: category + "/" + 'Tot_Occp_Units_CEN_2010' }, inplace = True )

    del dataframe['Tot_Occp_Units_CEN_2010']
    #del dataframe['Tot_Housing_Units_CEN_2010'] #not really important.
    #for some reason, there are some places that just suck at reporting good data.
    dataframe.fillna( 0, inplace = True )    
    
    dataframe = generate_features( dataframe )
    return dataframe




      
def cv( model, data, response, weights, K=5, location_data = [], report_training=True):
    """
    model is the scikitlearn model, with parameters already set. This is really to restrictive, eg: if I want to do preprocessing on the 
    cv'ed set, there is no good way.
    """
    kf = KFold( len(response), K , indices = False, shuffle=True) 
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
	

    	print "CV %i: Test accuracy: %s" % (i, create_confidence_interval( cv_scores[:i], 0.95))        
    	if report_training:
            n_testing_data = testing_response.shape[0]
            try:
                    training_location[0] = training_location[0][:n_testing_data] 
            except:
                    pass
            
            predict_args = [ training_data[:n_testing_data] ] + training_location        
            training_scores[i-1] = WMAEscore( model.predict( *predict_args  ), training_response[:n_testing_data], training_weights[:n_testing_data])
            print "CV %i: Train accuracy: %s" % (i, create_confidence_interval(training_scores[:i], 0.95) )
    	print "--------------------------------"
    	i += 1 


    print "Test accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() / 2)
    
    if report_training:
        print "Train accuracy: %0.2f (+/- %0.2f)" % (training_scores.mean(), training_scores.std() / 2)

    
def create_confidence_interval( array, alpha):
     
     td = stats.t
 
     n = array.shape[0]
     mu = array.mean()
     ci_width = td.interval(alpha, n)[1]*array.std()/np.sqrt(n)	
     
     return "%0.3f ( +- %0.3f)"%(mu, ci_width)	
        
def find_geo_NN( lat, long, location_data, K = 1 ):
    #location_data is a 2-d nx2 numpy array of lat-long coordinates.
    v =  (( location_data - np.array( [lat, long] )  )**2).sum(axis=1)
    ix = bn.argpartsort( v, K, axis=None)[:K]
    ix2 =ix[ np.nonzero( v[ix] < 100)]
    ix2 = np.append( ix2, np.random.randint(0, location_data.shape[0], (1, K - ix2.shape[0] )  )  )
    return ix2
    

    

    
