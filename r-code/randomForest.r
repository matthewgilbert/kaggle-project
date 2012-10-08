#R --no-save <~/kaggle-project/r-code/randomForest.r > RFSeqlog.out 2>&1 &

####Fit RF Model############
library(randomForest)
load("cleaned.dat")
index_train = grep("GIDBG|LATITUDE|LONGITUDE|weight", names(census.formatted.df), invert=TRUE)
data = census.formatted.df[,index_train]
fit.rf <- randomForest(Mail_Return_Rate_CEN_2010 ~ ., data = data, ntree=250 )

#####File for analysis ########
save(fit.rf, file='rfSeq.RData')
