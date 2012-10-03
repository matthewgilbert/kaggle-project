###################################
# Data loading
###################################

#setwd("C:\\Documents and Settings\\Administrator\\My Documents\\Dropbox\\census")
setwd("/home/matthew/Git/kaggle-project/data")

census.df <- read.csv(file = "training_filev1.csv")


###################################
# Data Preprocessing
###################################

#85 corresponds to non english euro area, 15 is total ACS survey population
percent_nonenglish = census.df[,91]/census.df[,15]

#83 corresponds to spanish area, 15 is total ACS survey population
percent_spanish = census.df[,83]/census.df[,15]

percent_mobile = census.df[,151]/census.df[,130]

percent_urban = census.df[,11]/census.df[,14]
percent_rural = census.df[,13]/census.df[,14]

percent_old = census.df[,68]/census.df[,15]

#appears to be a downward trend
percent_poor = census.df$Prs_Blw_Pov_Lev_ACS_06_10/census.df$Tot_Population_ACS_06_10

percent_educated = census.df$College_ACS_06_10/census.df$Tot_Population_ACS_06_10

#the percentage of spanish ballots mailed out, 167 is total number of ballots, 170 is spanish ballots
spanishBallots = census.df[,170]
spanishBallots[is.na(spanishBallots)] = 0
percent_spanishBallots = spanishBallots/census.df[,167]

#ballot difference
ballot_diff = percent_spanish - percent_spanishBallots

#171 refers to return rate category
rate = census.df[,171]

plot(percent_poor, rate,pch='.')

#plot
require('ggplot2')
test.df = read.csv('training_file_plus_location.csv')
loc = test.df[,c(174,173,171)]
index = which(loc$Mail_Return_Rate_CEN_2010 < 60)
loc[index,3] = 60
qplot(loc$LONGITUDE,loc$LATITUDE,data=loc, colour=loc$Mail_Return_Rate_CEN_2010, size='.') + scale_color_gradient(low='blue', high='red')


##simple error calculation of using y_hat = E[Ret_Rate] and y_hat = E[Ret_Rate_weighted] trained on 1/4 of training data and all training data
weights = census.formatted.df$weight
Y = census.formatted.df$Mail_Return_Rate_CEN_2010
y_hat1 = mean(Y)
error1 = 1/sum(weights)*sum( weights*abs(Y - y_hat1) )

partition1 = 1:floor(length(Y)/4)
partition2 = (floor(length(Y)/4)+1):length(Y)
y_hat2 = mean(Y[partition1])
error2 = 1/sum(weights[partition2])*sum( weights[partition2]*abs(Y[partition2] - y_hat2) )

#simple weighted regression on the data, not this data likely includes redundancies since ACS and non ACS data are included
data = census.formatted.df[,(1:101)]
weights = census.formatted.df$weight

# random forest
##############################################
library(randomForest)
load("cleaned.dat")
#memory.limit(4000)

data = census.formatted.df[,(1:101)]

index = sample(1:nrow(data), 5000)
subsample = data[index,]
test_index = setdiff(1:nrow(data),index)
testsample = data[test_index,]

# can't run random forest, out of memory
fit.rf <- randomForest(Mail_Return_Rate_CEN_2010 ~ ., data = subsample )

predictions = predict(fit.rf, testsample)

weights = census.formatted.df$weight[test_index]
Y = testsample[,101]
error.rf = 1/sum(weights)*sum( weights*abs(Y - predictions) )

###trained on full data set
library(randomForest)
load("cleaned.dat")
data = census.formatted.df[,(1:102)]
fit.rf <- randomForest(Mail_Return_Rate_CEN_2010 ~ ., data = data )

#Error: cannot allocate vector of size 988.1 Mb
