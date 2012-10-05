clean <- function(series) {

    if (series != 'training' && series != 'test')
        stop('Please input either training or test as the series')

    #Data Loading and Cleaning
    census.df <- read.csv(file = paste(series, "_filev1.csv", sep=''))

    temp.df = census.df

    index = seq(1,length(names(census.df)))

    #lists of indices to omit, groupped according to different metrics
    geography_omit = c(1,2,3,4,5,6,7,8)
    error_omit = grep("MOE",names(census.df))
    na_indices = c(168,169)
    na_zero_index = 170

    #change all money indices to numeric
    money_index = c(124,125,126,127,128,129,161,162,163,164,165,166)
    for(i in money_index) {
        temp.df[,i] = as.numeric(temp.df[,i])
    }
    #change places which received no English/Spanish ballots from NA to 0
    temp.df[is.na(temp.df[,na_zero_index]),na_zero_index] = 0

    index = setdiff(setdiff(setdiff(index,geography_omit),error_omit),na_indices)
    census.cleaned.df = temp.df[,index]


    #Data Transformation --> changing raw numbers to percentages

    census.formatted.df = census.cleaned.df
    unformatted_index = 1:length(census.formatted.df)
    #non formattable consists of land area stats and median house value stats, return rate and weights
    non_formattable = c(1,2,75,76,96,97,101,102)
    pop_index = c(3,4,5)
    totalPop_index = 6
    acs_totalPop_index = 7
    acs_totalHouse_index = 99

    unformatted_index = setdiff(setdiff(setdiff(setdiff(unformatted_index,non_formattable),totalPop_index),acs_totalPop_index),acs_totalHouse_index)

    census.formatted.df[,pop_index] = census.formatted.df[,pop_index]/census.formatted.df[,totalPop_index]
    unformatted_index = setdiff(unformatted_index,pop_index)

    #get all population acs stats
    acs_index = intersect(grep("ACS",names(census.formatted.df)), unformatted_index)
    acs_pop_index = acs_index[(acs_index <= 49)]
    acs_house_index = setdiff(acs_index, acs_pop_index)

    no_pop_index = which(census.formatted.df[,acs_totalPop_index] == 0)
    census.formatted.df[no_pop_index, acs_totalPop_index] = Inf    
#    keep = setdiff(1:nrow(census.formatted.df), no_pop_index)
#    census.formatted.df = census.formatted.df[keep,]
    

    census.formatted.df[,acs_pop_index] = census.formatted.df[,acs_pop_index] / census.formatted.df[,acs_totalPop_index]
    unformatted_index = setdiff(unformatted_index,acs_pop_index)

    census.formatted.df[,acs_house_index] = census.formatted.df[,acs_house_index] / census.formatted.df[,acs_totalHouse_index]
    unformatted_index = setdiff(unformatted_index,acs_house_index)

    spanish_ballot_index = 100
    census.formatted.df[,spanish_ballot_index] = census.formatted.df[,spanish_ballot_index] / census.formatted.df[,acs_totalHouse_index]
    unformatted_index = setdiff(unformatted_index,spanish_ballot_index)

    #I believe that acs vs. non acs total housing is the same since the acs refers to sampling period
    non_acs_housing = unformatted_index[(unformatted_index > 39)]
    census.formatted.df[,non_acs_housing] = census.formatted.df[,non_acs_housing] / census.formatted.df[,acs_totalHouse_index]
    unformatted_index = setdiff(unformatted_index,non_acs_housing)

    nonACS_pop_index = unformatted_index
    census.formatted.df[,nonACS_pop_index] = census.formatted.df[,nonACS_pop_index] / census.formatted.df[,totalPop_index]

    #hack to add in geo coordinates
    locations <- read.csv(file = paste(series, "_locations.csv", sep=''))
    census.formatted.df = cbind('LATITUDE'=locations[,1], 'LONGITUDE'=locations[,2], census.formatted.df)
    #hack to add back in GidBG so don't have to change all the hardcoded indices above
    census.formatted.df = cbind('GIDBG'=census.df$GIDBG, census.formatted.df)

    #remove redundant categories, e.g. don't need number of males and number of females
    redundant_index = grep("Females_CEN_2010|Females_ACS_06_10",names(census.formatted.index),invert=TRUE)

    if (series == 'training') {
        save(census.df, census.cleaned.df, census.formatted.df, file = 'cleaned.dat')
        write.table(census.formatted.df,"formattedData.csv")
        sprintf("The file cleaned.dat and formattedData.csv were printed to %s", getwd())
    } else if (series == 'test') {
        test.census.df = census.df
        test.census.cleaned.df = census.cleaned.df
        test.census.formatted.df = census.formatted.df
        save(test.census.df, test.census.cleaned.df, test.census.formatted.df, file = 'cleanedtest.dat')
        write.table(test.census.formatted.df,"formattedDatatest.csv")
        sprintf("The file cleanedtest.dat and formattedDatatest.csv were printed to %s", getwd())
    }

}

