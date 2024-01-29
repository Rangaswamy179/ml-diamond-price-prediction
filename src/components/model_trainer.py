import os,sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from dataclasses import dataclass

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
  trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config=ModelTrainerConfig()

  def initiate_model_trainer(self,train_arr,test_arr):
    try:
      logging.info('splitting dependent and independent train and test arrays')
      x_train,y_train,x_test,y_test=(
        train_arr[:,:-1],
        train_arr[:,-1],
        test_arr[:,:-1],
        test_arr[:,-1]
      )
      models={
        'Linearregression':LinearRegression(),
        'Lasso':Lasso(),
        'Ridge':Ridge(),
        'Elasticnet':ElasticNet(),
        'Decisiontree':DecisionTreeRegressor(),
        'RandomForest':RandomForestRegressor(),
        'Knearest nighbour':KNeighborsRegressor()
      }

      model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,models)
      print(model_report)
      print('\n------------------------------------------------\n')
      logging.info(f"model report :{model_report}")

      best_model_score=max(sorted(model_report.values()))

      best_model_name=list(model_report.keys())[
        list(model_report.values()).index(best_model_score)
      ]

      best_model=models[best_model_name]

      print(f'Best model found ,model_name:{best_model_name},R2 score:{best_model_score}')
      print('\n-------------------------------------------\n')
      logging.info(f'Best model found ,model_name:{best_model_name},R2 score:{best_model_score}')

      save_object(
        file_path=self.model_trainer_config.trained_model_file_path,
        obj=best_model
      )
      


    except Exception as e:
      raise CustomException(e,sys)
