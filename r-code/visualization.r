###################################
# Data loading
###################################

setwd("C:\\Documents and Settings\\Administrator\\My Documents\\Dropbox\\census")
#setwd("/home/matthew/Git/kaggle-project/data")

census.df <- read.csv(file = "training_filev1.csv")


###################################
# Data Preprocessing
###################################

library(maps)
library(mapproj)
data(county.fips)

my_county.fips = as.character(census.df$GIDBG)
fips.length = nchar(my_county.fips)

fips = as.numeric(substring(my_county.fips, 1, fips.length - 7))
unique.fips = unique(fips)

# declare a new data.frame collapse the rate on a county level.
county_level_average <- unique.fips

for(i in 1:length(unique.fips)){
	index = which(unique.fips[i] == fips)
	county_level_average[i] = mean(census.df[index, 171], na.rm = T)
}


###################################
# Plotting
###################################
category <- as.numeric(cut(county_level_average, c(seq(50,100,by=10),max(county_level_average))))
cols <- c("#F1EEF6", "#D4B9DA", "#C994C7", "#DF65B0", "#DD1C77", "#980043")

pdf(file = "county_thematic.pdf")
map("county", col=cols[category[match(county.fips$fips,unique.fips)]], fill = TRUE,resolution = 0,lty = 0,projection = "polyconic")
map("state",col = "white",fill=FALSE,add=TRUE,lty=1,lwd=1,projection="polyconic")
dev.off()


