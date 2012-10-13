#run on server with
#R --no-save <~/kaggle-project/r-code/randomForestLocalCoarse.r > RFLocalLogCoarse.out 2>&1 &

####Fit RF Model############
library(randomForest)
library(foreach)
library(doMC)

coreUse = 42
registerDoMC(coreUse)
load("cleaned.dat")
load("cleanedtest.dat")

#format distance data
index_train = grep("GIDBG|LATITUDE|LONGITUDE|weight", names(census.formatted.df), invert=TRUE)
train_data = census.formatted.df[,index_train]
index_test = grep("GIDBG|LATITUDE|LONGITUDE|weight", names(test.census.formatted.df), invert=TRUE)
test_data = test.census.formatted.df[,index_test]
train_locations = data.frame("LATITUDE"=census.formatted.df$LATITUDE, "LONGITUDE"=census.formatted.df$LONGITUDE)
test_locations = data.frame("LATITUDE"=test.census.formatted.df$LATITUDE, "LONGITUDE"=test.census.formatted.df$LONGITUDE)

#Calculate the closest indices
source("~/kaggle-project/r-code/distances.r")
numNearest = 1000
distance.info = DistanceInfo(train_locations, test_locations, numNearest, coreUse)
closest_indices = distance.info[[1]]

#train local random forests and make predictions base on these
predictions = data.frame("x"=rep(NA,nrow(test_locations)))
int_predictions = rep(NA,partition)
predictions <- foreach(index = (1:coreUse), .combine=c, .packages='randomForest') %dopar% {
                j = 1
                lower_lim = partition*(index-1)+1
                upper_lim = partition*index
                for (i in (lower_lim:upper_lim)) {        
                    fit.rf <- randomForest(Mail_Return_Rate_CEN_2010 ~ ., data=train_data[closest_indices[i,],])
                    int_predictions[j] = predict(fit.rf, test_data[i,])
                    j = j+1
                }
                predictions = int_predictions
}
proc.time()

####Write predictions submission file#######
dir.create('Results')
write.table(predictions,"Results/RF-PredictionsLocal.csv",row.names = FALSE)



