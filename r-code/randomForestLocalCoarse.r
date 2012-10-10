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


index_train = grep("GIDBG|LATITUDE|LONGITUDE|weight", names(census.formatted.df), invert=TRUE)
train_data = census.formatted.df[,index_train]

index_test = grep("GIDBG|LATITUDE|LONGITUDE|weight", names(test.census.formatted.df), invert=TRUE)
test_data = test.census.formatted.df[,index_test]

train_locations = data.frame("LATITUDE"=census.formatted.df$LATITUDE, "LONGITUDE"=census.formatted.df$LONGITUDE)
test_locations = data.frame("LATITUDE"=test.census.formatted.df$LATITUDE, "LONGITUDE"=test.census.formatted.df$LONGITUDE)

#convert to cartesian locations coordinates
train_Latitude = train_locations$LATITUDE/180*pi
train_Longitude = train_locations$LONGITUDE/180*pi
test_Latitude = test_locations$LATITUDE/180*pi
test_Longitude = test_locations$LONGITUDE/180*pi
r = 6371
x_train = r*cos(train_Longitude)*cos(train_Latitude)
y_train = r*cos(train_Longitude)*sin(train_Latitude)
z_train = r*sin(train_Longitude)
x_test = r*cos(test_Longitude)*cos(test_Latitude)
y_test = r*cos(test_Longitude)*sin(test_Latitude)
z_test = r*sin(test_Longitude)

xyz_trainLocations = data.frame("x"=x_train, "y"=y_train, "z"=z_train)
xyz_testLocations = data.frame("x"=x_test, "y"=y_test, "z"=z_test)

predictions = data.frame("x"=rep(NA,nrow(xyz_testLocations)))
#define the number of surroundings we will look at
nearest = 1000

#Note, if bins/coreUse is not an int spurious behavior may arise
bins = nrow(test_data)
partition = bins/coreUse

int_closest_indices = matrix(NA,partition,nearest)
closest_indices <- foreach(index=(1:coreUse), .combine=rbind) %dopar% {
                    j = 1
                    lower_lim = partition*(index-1)+1
                    upper_lim = partition*index
                    for(i in (lower_lim:upper_lim)) {
                        d = sqrt((x_train-xyz_testLocations[i,1])^2+(y_train-xyz_testLocations[i,2])^2+(z_train-xyz_testLocations[i,3])^2)
                        int_closest_indices[j,] = order(d)[1:nearest]
                        j = j+1
                    }
                    closest_indices = int_closest_indices
}


int_predictions = rep(NA,partition)
predictions <- foreach(index = (1:coreUse), .combine=c, .packages='randomForest') %dopar% {
                j = 1
                lower_lim = partition*(index-1)+1
                upper_lim = partition*index
                for (i in (lower_lim:upper_lim) {        
                    fit.rf <- randomForest(Mail_Return_Rate_CEN_2010 ~ ., data=train_data[closest_indices[i,],])
                    int_predictions[j] = predict(fit.rf, test_data[i,])
                    j = j+1
                }
                predictions = int_predictions
}
proc.time()

####Write predictions submission file#######
dir.create('Results')
write.table(predictions,"Results/RF-PredictionsParallel.csv",row.names = FALSE)



