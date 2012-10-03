####Fit RF Model############
library(randomForest)
load("cleaned.dat")
index_train = (names(census.formatted.df) != "weight")
data = census.formatted.df[,index_train]
fit.rf <- randomForest(Mail_Return_Rate_CEN_2010 ~ ., data = data )

####Make Predictions#########
load("cleanedtest.dat")
index_test = (names(test.census.formatted.df) != "weight")
test.data = test.census.formatted.df[,index_test]
predictions = predict(fit.rf, test.data)

####Write predictions submission file#######
write.table(predictions,"RF-Predictions.csv",row.names = FALSE)
