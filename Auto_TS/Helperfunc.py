import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from auto_ts import auto_timeseries
import pathlib
import os

def fileType(filePath) : 
  extension = pathlib.Path(filePath).suffix
  if extension != '.csv' and extension != '.xlsx' :
    extension = "invalid"
  return extension

def parse(filePath) : 
  extension = fileType(filePath);
  if(extension == "invalid") : 
    print("Sorry! File format not supported. Try excel or csv") 
  if extension == ".xlsx" :
    return pd.read_excel(filePath);
  else :
    return pd.read_csv(filePath);


def setIndex(var,train_df,test_df):
  train_df.set_index(var, inplace=True)
  test_df.set_index(var, inplace=True)
  train_df.index = pd.to_datetime(train_df.index)
  test_df.index = pd.to_datetime(test_df.index)

def PlotGraph(titleTrain, titleTest,figSize,y):
  train_df[y].plot(figsize=figSize, title= titleTrain, fontsize=14)
  test_df[y].plot(figsize=figSize, title= titleTest, fontsize=14)
  plt.show()

def GetModel(period, score, model, interval, tcColumn, target,train_df):
  # Genrating auto timeseries modal
  model = auto_timeseries(forecast_period=period,
                          score_type=score, time_interval=interval,
                          model_type=model)
  
  # Reseting the index of the train data so the data is ordered
  train_df.reset_index(inplace=True)
  
  # fitting the model i.e genrating weights regarding the model
  leaderboard, best_name = model.fit(traindata= train_df, 
            ts_column=tcColumn,
            target=target,
            cv=5)
  
  return model,leaderboard, best_name

# Creating a prediction pipeline
def predictionpipeline(model,test_df,name):
  future_predictions = model.predict(test_df)
  test_df.reset_index(inplace=True)    # reseting the testdata according to dates in assending order
  test_df[name] = future_predictions["yhat"].values    # Creating a new column in the test dataFrame with the name "Prophet Predictions" and assigning it the predicted values of the model

if __name__=="__main__":
    BASE_DIR= os.getcwd()
    
    train_file = os.path.join(BASE_DIR,'Dataset/Train.csv')
    test_file = os.path.join(BASE_DIR,'Dataset/Test.csv')
    train_df = parse(train_file)
    test_df = parse(test_file)
    print(test_df.head())
    setIndex("dteday",train_df,test_df)
    model,leaderboard, best_name = GetModel(61,'rmse','best','D','dteday', 'cnt',train_df)
    print(leaderboard,"\n\n",best_name)
    predictionpipeline(model,test_df, "Prophet Predictions")
    print(test_df.head())