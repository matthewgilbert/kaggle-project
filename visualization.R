setwd("C:\\Documents and Settings\\Administrator\\My Documents\\Dropbox\\census")

census.df <- read.csv(file = "training_filev1.csv")

library(maps)
data(county.fips)

my_county.fips = as.character(census.df$GIDBG)
fips.length = nchar(my_county.fips)

fips = as.numeric(substring(my_county.fips, 1, fips.length - 7))
unique.fips = unique(fips)

# declare a new data.frame collapse the rate on a county level.
county_level_average <- unique.fips

for(fip in unique.fips){
	index = which(fip in my_county.fips)
	
}