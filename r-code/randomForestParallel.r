#run on server with
#R --no-save <~/kaggle-project/r-code/randomForestParallel.r > RFParallelLog.out 2>&1 &

####Fit RF Model############
library(randomForest)
library(foreach)
library(doMC)

registerDoMC(32)
load("cleaned.dat")
index_train = grep("GIDBG|LATITUDE|LONGITUDE|weight", names(census.formatted.df), invert=TRUE)
data = census.formatted.df[,index_train]

fit.rf <- foreach(ntree=rep(16, 32), .combine=combine, .packages='randomForest') %dopar% {
    randomForest(Mail_Return_Rate_CEN_2010 ~ ., data=data, ntree=ntree )
}
proc.time()
####Make Predictions#########
load("cleanedtest.dat")
index_test = grep("GIDBG|LATITUDE|LONGITUDE|weight", names(test.census.formatted.df), invert=TRUE)
test.data = test.census.formatted.df[,index_test]
predictions = predict(fit.rf, test.data)
proc.time()
####Write predictions submission file#######
dir.create('Results')
write.table(predictions,"Results/RF-PredictionsParallel.csv",row.names = FALSE)
save(fit.rf, file="Results/rfParallel.RData")
