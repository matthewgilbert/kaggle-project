DistanceInfo <- function(train_locations, test_locations, numNearest, coreUse) {

    #STILL NEEDS TO BE TESTED
    #Takes in the locations of the training data, locations of the testing data,
    #the number of points in a neighborhood to look at, the number of cores to use

    library(foreach)
    library(doMC)

    #convert to cartesian locations coordinates
    train_Latitude = train_locations$LATITUDE/180*pi
    train_Longitude = train_locations$LONGITUDE/180*pi
    test_Latitude = test_locations$LATITUDE/180*pi
    test_Longitude = test_locations$LONGITUDE/180*pi
    #radius of the earth
    r = 6371
    x_train = r*cos(train_Longitude)*cos(train_Latitude)
    y_train = r*cos(train_Longitude)*sin(train_Latitude)
    z_train = r*sin(train_Longitude)
    x_test = r*cos(test_Longitude)*cos(test_Latitude)
    y_test = r*cos(test_Longitude)*sin(test_Latitude)
    z_test = r*sin(test_Longitude)

    xyz_trainLocations = data.frame("x"=x_train, "y"=y_train, "z"=z_train)
    xyz_testLocations = data.frame("x"=x_test, "y"=y_test, "z"=z_test)

    #Note, if bins/coreUse is not an int spurious behavior may arise
    bins = nrow(test_locations)
    partition = bins/coreUse

    int_closest_indices = matrix(NA,partition,nearest)
    int_max_distance = rep(NA, partition)
    closest_indices <- foreach(index=(1:coreUse), .combine=rbind) %dopar% {
                        j = 1
                        lower_lim = partition*(index-1)+1
                        upper_lim = partition*index
                        for(i in (lower_lim:upper_lim)) {
                            d = sqrt((x_train-xyz_testLocations[i,1])^2+(y_train-xyz_testLocations[i,2])^2+(z_train-xyz_testLocations[i,3])^2)
                            int_closest_indices[j,] = order(d)[1:nearest]
                            int_max_distance[j] = sort(d, decreasing=TRUE)[1]
                            j = j+1
                        }
                        closest_indices = cbind(int_closest_indices, max_distances)
    }
    DistanceInfo = list(closest_indices[,1:(ncol(closest_indices)-1)], closest_indices[,ncol(closest_indices)])

}
