#run on server with
#R --no-save <~/kaggle-project/r-code/condRF.r > RFCondVarImp.out 2>&1 &

####Fit RF Model############
library(party)

load("cleaned.dat")
index_train = grep("GIDBG|LATITUDE|LONGITUDE|weight", names(census.formatted.df), invert=TRUE)
data = census.formatted.df[,index_train]

fit.rf <- cforest(Mail_Return_Rate_CEN_2010 ~ ., data=data)
proc.time()
####Make Predictions#########
load("cleanedtest.dat")
index_test = grep("GIDBG|LATITUDE|LONGITUDE|weight", names(test.census.formatted.df), invert=TRUE)
test.data = test.census.formatted.df[,index_test]
predictions = predict(fit.rf, test.data)
varimp(fit.rf)
####Write predictions submission file#######
dir.create('Results')
write.table(predictions,"Results/RF-CondPredictions.csv",row.names = FALSE)
save(fit.rf, file="Results/rfCond.RData")
