rm(list = ls())
options(warn=-1)

library(fpp)
library(dplyr)
library(Metrics)
library(randomForest)
library(xgboost)
#### Call Chocolate sales for 2 years.csv
inputData = read.csv("Sales_Final.csv",stringsAsFactors=FALSE)
colnames(inputData) = c("Date","Product", "Store", "Qty","Sales","Inventory", "Promotion", "AvgQty")
inputData$Date = as.Date(inputData$Date,format = "%d-%m-%Y")
inputData$Qty <- as.numeric(inputData$Qty)
inputData$Promotion <- ifelse(inputData$Promotion=="Y",1,0)

#102443 615783  758078
# Taking only 1st product...
inputData <- arrange(filter(inputData, Product=="758078"),Date)
inputData <- subset(inputData,inputData$Date >="2015-01-01" & inputData$Date <="2016-12-07")
inputData <- select(inputData, Date, Qty, Promotion)


Lotto = read.csv("Lotto_Final.csv",stringsAsFactors = F)
Lotto$Date = as.Date(Lotto$Date,format = "%d-%b-%y")
Lotto <- subset(Lotto,Lotto$Date >="2015-01-01" & Lotto$Date <="2016-12-07")
Lotto$NewSwissLottoAmount <-  na.locf(Lotto$Swiss_Lotto_Amount,fromLast=TRUE)
Lotto$NewJoker_Amount <-  na.locf(Lotto$Joker_Amount,fromLast=TRUE)
Lotto$NewEuro_Millions_Amount <-  na.locf(Lotto$Euro_Millions_Amount,fromLast=TRUE)

#write.csv(Lotto,"NewLotto.csv")
LottoVar <- c("Date", "Swiss_Lotto","NewSwissLottoAmount","Joker","NewJoker_Amount","Euro_Millions", "NewEuro_Millions_Amount")
Lotto <- Lotto[LottoVar]
rm(LottoVar)


#### Prepartion of Holiday Data

Holiday = read.csv("Final_Holiday_New.csv",stringsAsFactors = F,sep="|")
HolidayVar  <-  c("DATE_CY","Is_Holiday")
Holiday  <- Holiday[HolidayVar]
Holiday$DATE_CY = as.Date(Holiday$DATE_CY)
Holiday <- subset(Holiday,Holiday$DATE_CY >="2015-01-01" & Holiday$DATE_CY <="2016-12-07")
Holiday$Is_Holiday <- ifelse(Holiday$Is_Holiday=="Y",1,0)
names(Holiday) <- c("Date","Holiday")
rm(HolidayVar)
#write.csv(Holiday,"ListHoliday.csv")


#### Preparation of Weather Data

Weather = read.csv("Luzen.csv",stringsAsFactors = F)
Weather$Date = as.Date(Weather$Date,format = "%m/%d/%Y")
Weather <- subset(Weather,Weather$Date >="2015-01-01" & Weather$Date <="2016-12-07")


### merge Lotto , Holiday ,Weather and InputData

FinalData = merge(inputData,Lotto,by="Date",all.x = T)
FinalData = merge(FinalData,Holiday,by="Date",all.x = T)
FinalData = merge(FinalData,Weather,by="Date",all.x = T)

rm(inputData,Holiday,Lotto,Weather)
#-----------------------------------------------------------------------------------------------

#### get Month and Day Parameter

FinalData$Day = as.numeric(format(FinalData$Date,"%d"))
FinalData$Month = as.numeric(format(FinalData$Date,"%m"))
FinalData$Year = as.numeric(format(FinalData$Date,"%Y"))

####### Final data frame (FinalData) to use above Train+Validation+Test

FinalData$NewJoker_Amount = log(FinalData$NewJoker_Amount)

Training = subset(FinalData, Date >= "2015-01-01" & Date <= "2016-06-15")
Validation = subset(FinalData, Date >= "2016-06-16" & Date <= "2016-11-22")
Test = subset(FinalData, Date >= "2016-11-23" & Date <= "2016-12-07")

Training = subset(Training,select = -c(Date,Max.Gust.SpeedKm.h,Events,CloudCover,Max.VisibilityKm,Mean.VisibilityKm,Min.VisibilitykM))
Validation = subset(Validation,select = -c(Date,Max.Gust.SpeedKm.h,Events,CloudCover,Max.VisibilityKm,Mean.VisibilityKm,Min.VisibilitykM))
Test = subset(Test,select = -c(Date,Max.Gust.SpeedKm.h,Events,CloudCover,Max.VisibilityKm,Mean.VisibilityKm,Min.VisibilitykM))



RMPSE<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab   <-  labels
  epreds <-  preds
  err    <- sqrt(mean(((elab - epreds)/elab)^2))
  return(list(metric = "RMPSE", value = err))
}

feature.names <- names(Training)  # remove (Sales), use colnames()
feature.names = feature.names[2:length(feature.names)]
#feature.names = feature.names[c(1:3,5,8,10,25:27)]
tra<-Training[,feature.names]
valid = Validation[,feature.names]

dval<-xgb.DMatrix(data=data.matrix(valid),label=Validation$Qty)
dtrain<-xgb.DMatrix(data=data.matrix(tra),label=Training$Qty)

watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear", 
                booster = "gbtree",
                eta = 0.01, # 0.06, #0.01,
                gamma = 0.1,
                min_child_weight = 0.1,
                max_depth           = 15, #changed from default of 8
                subsample           = 0.7, # 0.7
                colsample_bytree    = 0.7, # 0.7
                num_parallel_tree   = 5,
                alpha = 0.0001, 
                lambda = 1
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 7000, #300, #280, #125, #250, # changed from 300
                    verbose             = 0,
                    early.stop.round    = 200,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=RMPSE
)

TestPred = predict(clf, data.matrix(Test[,feature.names]))

RMPSE.normal<- function(preds, actual) {
  err    <- sqrt(mean(((actual - preds)/actual)^2))
  return(list(metric = "RMPSE", value = err))
}

rmpse.test = RMPSE.normal(TestPred,Test$Qty)
mape.test = round(mean(abs((Test$Qty-TestPred)*100/Test$Qty)),2)
mpe.test = round(mean((Test$Qty-TestPred)*100/Test$Qty),2)

print(paste("RMPSE of XGB model on Test set last 15 days : ", rmpse.test$value,"%"))
print(paste("MAPE of XGB model on Test set last 15 days : ", mape.test,"%"))
print(paste("MPE of XGB model on Test set last 15 days : ", mpe.test,"%"))
options(warn=0)

