## Data loading
load('cleaned.dat')

#plot
require('ggplot2')
category.index = grep("LATITUDE|LONGITUDE|Mail_Return_Rate", names(census.formatted.df)) 
loc = census.formatted.df[,category.index]
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
