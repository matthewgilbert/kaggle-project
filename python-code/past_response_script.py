import numpy as np
import pandas 


def sf(x):
    try:
        return  "{0:011d}".format(int(x))
    except:
        return np.nan


raw_data = pandas.read_csv( "../test_file_plus_location.csv")
past_response = pandas.read_csv( "../FriendlyParticipationRates.csv", delimiter="\t" )
print "data in"
#remove some nan


past_response['Concatenated GEO ID'] = past_response['Concatenated GEO ID'].map( sf )

raw_data['GIDBG']  = raw_data['GIDBG'].map( lambda x: "%s"%( str(x)[:-1] ).zfill(11) )
raw_data['2000_response'] = np.empty( raw_data.shape[0] ).fill( np.nan )
print "data fed and ready"
count = 0

for i, rd in enumerate( raw_data.GIDBG ):
    for j,s in enumerate( past_response['Concatenated GEO ID']):
        if s==rd:
            count += 1
            raw_data['2000_response'][i] = past_response['2000 Participation Rate'][j]
            print i
            
print "Filled in: %d"%count

raw_data.to_csv("../test_file_plus_loc_and_response.csv", sep = "," )
print "Done"
