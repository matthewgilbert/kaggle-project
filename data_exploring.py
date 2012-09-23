"""
Kaggle Census competition
"""



import pandas #good data tables, like R dataframes
import census_utilities 



data = pandas.read_csv( "training_filev1.csv")
response = data['Mail_Return_Rate_CEN_2010']
weights = data['weight']



#Some of the data is not really useful in a regression (state_id, state_name etc), 
# so lets delete it. Also remove the response variable.
to_remove = ['Mail_Return_Rate_CEN_2010', 'State', 'State_name', 'County_name', 
             'County', 'GIDBG', 'Tract', 'Block_Group', 'Flag', 'weight']
for to_rm in to_remove:
    del data[to_rm]

    
to_convert_to_float = ['Med_HHD_Inc_BG_ACS_06_10', 'Med_HHD_Inc_BG_ACSMOE_06_10', 'Med_HHD_Inc_TR_ACS_06_10', 'Med_HHD_Inc_TR_ACSMOE_06_10',
 'Aggregate_HH_INC_ACS_06_10', 'Aggregate_HH_INC_ACSMOE_06_10', 'Med_House_Val_BG_ACS_06_10','Med_House_Val_BG_ACSMOE_06_10''Med_house_val_tr_ACS_06_10',
'Med_house_val_tr_ACSMOE_06_10', 'Aggr_House_Value_ACS_06_10', 'Aggr_House_Value_ACSMOE_06_10']

for to_cn in to_convert_to_float:
    data[to_cn] = data[to_cn].map( census_utilities.money2float )
    

#do a naive filling of missing data
data.fillna( method="bfill", inplace=True)
data.fillna( method="ffill", inplace=True)


#Ok data is pretty clean, what can we do with it?


