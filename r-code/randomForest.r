####Fit RF Model############
library(randomForest)
load("cleaned.dat")
index = (names(census.formatted.df) != "weight")
data = census.formatted.df[,index]
fit.rf <- randomForest(Mail_Return_Rate_CEN_2010 ~ ., data = data )

####Make Predictions#########
load("cleanedtest.dat")
test.data = test.census.formatted.df[,index]
predictions = predict(fit.rf, test.data)

####Write predictions submission file#######
write.table(predictions,"RF-Predictions.csv")
