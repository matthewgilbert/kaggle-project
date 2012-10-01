#this script appends the location of the block to the end of the data piece it belongs to. 

import csv
import numpy as np






def find_long_lat( id, location_data ):
    """
    id is some GIDBG.
    return the (long, lat) Important, they are stringss
    """
    id = str(id)
    v = location_data[ location_data[ :,0] == id ]

    try:
        return [ v[0,2], v[0,3] ]
    except:
        id = "0"+id
        print id
        v = location_data[ location_data[ :,0] == id ]
        try:
                print "ok"
                return [ v[0,2], v[0,3] ]
        except:
            print "Exception: did not find %s."%id
            return [ np.NaN, np.NaN ]
        
        
        
        
        

write = open( '../formatted_training_file_plus_location.csv', mode = "wb" )
read = open( '../formatted.csv', mode = "r")


csv_write = csv.writer(write, delimiter="," )
csv_read = csv.reader(read, delimiter="," )

location_data =np.genfromtxt("../CEN2010Edited.csv", delimiter=",", skip_header=1, dtype="str")

csv_write.writerow ( csv_read.next() + ['LATITUDE', 'LONGITUDE'] )
i=0
for row in csv_read:
    i+=1
    csv_write.writerow( row + find_long_lat( row[0], location_data ) )
    
    
csv_write.close()
csv_read.close()
