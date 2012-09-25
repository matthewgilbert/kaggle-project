#Data Loading and Cleaning

setwd("/home/matthew/Git/kaggle-project/data")
census.df <- read.csv(file = "training_filev1.csv")

temp.df = census.df

index = seq(1,length(names(census.df)))

#lists of indices to omit, groupped according to different metrics
geography_omit = c(1,2,3,4,5,6,7,8)
error_omit = grep("MOE",names(census.df))
na_indices = c(168,169,170)

#change all money indices to numeric
money_index = c(124,125,126,127,128,129,161,162,163,164,165,166)
for(i in money_index) {
    temp.df[,i] = as.numeric(temp.df[,i])
}

index = setdiff(setdiff(setdiff(index,geography_omit),error_omit),na_indices)

census.cleaned.df = temp.df[,index]
save(census.df, census.cleaned.df, file = 'cleaned.dat')

