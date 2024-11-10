import sys
import os
from dataclasses import dataclass
from tkinter import Y
from src.exception import CustomException
from src.logger import logging
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting train and test input data")
            X_train,Y_train,X_test,Y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "AdaBoostRegressor": AdaBoostRegressor(),
            }

            '''
            HYPER PARAMETER TUNING

            It is the process of selecting the optimal values for a machine learning model's hyperparameters.
            These are settings that control the learning process of the model, 
            rather than being learned from the data itself.

            n_estimators = It represents the number of decision trees in the ensemble
            Ensemble Learning: These algorithms create a "forest"(ensemble) of decision trees, each trained on a random subset of the data. 
            n_estimators: Determines the size of this forest.
            
            criterion = It determines the function used to measure the quality of a split at each node of the tree

            learning rate = The learning rate is a hyperparameter that controls the step size taken during gradient descent.
            It determines how quickly the model's parameters are updated in response to the error gradient.

            subsample = It controls the fraction of observations (rows) used to train each individual tree in an ensemble.

            depth = It determines the maximum depth of the tree, which is the number of levels a tree can grow.

            iterations = 

            '''
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-Neighors Regressor":{
                    "n_neighbors": [3,5,7,9,11]
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report :dict = evaluate_model(X_train = X_train,Y_train = Y_train,X_test = X_test,Y_test = Y_test,models = models, params = params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            R2_score = r2_score(Y_test, predicted)
            return R2_score


        except Exception as e:
            raise CustomException(e,sys)