import csv


write = open("../CEN2010Edited.csv", mode = 'wb')
read = open("../CenPop2010_Mean_BG.csv", 'r')

csv_write = csv.writer( write, delimiter="," )
csv_read = csv.reader( read, delimiter="," )

csv_read.next() #skip the header

for row in csv_read:
    csv_write.writerow( [ row[0]+row[1]+row[2]+row[3], row[4], row[5], row[6] ] )
    
