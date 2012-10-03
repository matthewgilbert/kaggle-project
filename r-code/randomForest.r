####Fit RF Model############
library(randomForest)
load("cleaned.dat")
index_train = grep("GIDBG|weight", names(census.formatted.df), invert=TRUE)
data = census.formatted.df[,index_train]
fit.rf <- randomForest(Mail_Return_Rate_CEN_2010 ~ ., data = data )

####Make Predictions#########
load("cleanedtest.dat")
index_test = grep("GIDBG|weight", names(test.census.formatted.df), invert=TRUE)
test.data = test.census.formatted.df[,index_test]
predictions = predict(fit.rf, test.data)

####Write predictions submission file#######
write.table(predictions,"RF-Predictions.csv",row.names = FALSE)
