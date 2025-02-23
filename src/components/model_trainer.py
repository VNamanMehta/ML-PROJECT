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
                train_arr[:,:-1],# all rows and all columns except the last column
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
HYPERPARAMETER TUNING

Hyperparameters are parameters that define how a machine learning model is trained, rather than being learned from the data itself.  
They affect how the model generalizes to new data and can significantly impact performance.

Hyperparameter tuning is the process of selecting the **optimal** values for these parameters to improve the model’s predictive performance.

Below, we define a dictionary `params` that contains various hyperparameters for different machine learning models.
Each model has specific hyperparameters that influence how it learns patterns in the data.

---

# MACHINE LEARNING MODELS & THEIR PARAMETERS:

1. **Decision Tree**:
   - A decision tree is a flowchart-like structure where each internal node represents a decision based on a feature.
   - The tree is built by recursively splitting the data into subgroups.
   - Hyperparameter:  
     - `criterion`: Determines the function used to measure the quality of a split at each node.  
       - `squared_error`: Measures variance reduction in regression trees.  
       - `friedman_mse`: A modification of squared error, optimized for boosting models.  
       - `absolute_error`: Uses mean absolute deviation, making it more robust to outliers.  
       - `poisson`: Used for count data regression.

2. **Random Forest**:
   - An ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting.
   - Hyperparameter:  
     - `n_estimators`: Specifies the number of trees in the forest.  
       - Higher values provide more stability but increase computational cost.  
       - Values: [8, 16, 32, 64, 128, 256] → Increasing power of 2 for optimized performance.

3. **Gradient Boosting**:
   - A boosting algorithm that builds trees sequentially, with each tree correcting the errors of the previous one.
   - Hyperparameters:
     - `learning_rate`: Controls the contribution of each tree (smaller values require more trees).  
       - Values: [0.1, 0.01, 0.05, 0.001] → Lower values mean slower but more precise learning.
     - `subsample`: Controls the fraction of the training data used in each iteration.  
       - Values: [0.6, 0.7, 0.75, 0.8, 0.85, 0.9] → Helps in reducing overfitting.
     - `n_estimators`: Number of boosting iterations (more iterations mean better learning).  
       - Values: [8, 16, 32, 64, 128, 256].

4. **Linear Regression**:
   - A simple model that fits a linear equation to the data.
   - No hyperparameters are specified, as it typically doesn’t require tuning.

5. **K-Nearest Neighbors (KNN) Regressor**:
   - A non-parametric model that predicts the output based on the average of `k` nearest neighbors.
   - Hyperparameter:
     - `n_neighbors`: Specifies the number of nearest neighbors used for prediction.  
       - Values: [3, 5, 7, 9, 11] → More neighbors smooth the predictions.

6. **XGBoost Regressor**:
   - A highly efficient gradient boosting implementation designed for structured/tabular data.
   - Hyperparameters:
     - `learning_rate`: Controls the step size of updates (same as Gradient Boosting).  
       - Values: [0.1, 0.01, 0.05, 0.001].
     - `n_estimators`: Number of boosting rounds (same as Gradient Boosting).  
       - Values: [8, 16, 32, 64, 128, 256].

7. **CatBoost Regressor**:
   - A gradient boosting model specifically optimized for categorical data.
   - Hyperparameters:
     - `depth`: Maximum depth of trees (higher depth captures more complex patterns but risks overfitting).  
       - Values: [6, 8, 10].
     - `learning_rate`: Same as in other boosting algorithms.  
       - Values: [0.01, 0.05, 0.1].
     - `iterations`: Number of boosting rounds.  
       - Values: [30, 50, 100] → Higher iterations allow more learning.

8. **AdaBoost Regressor**:
   - A boosting technique that combines multiple weak learners into a strong model.
   - Hyperparameters:
     - `learning_rate`: Determines how much each weak learner contributes.  
       - Values: [0.1, 0.01, 0.5, 0.001].
     - `n_estimators`: Number of weak learners.  
       - Values: [8, 16, 32, 64, 128, 256].

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
            # Evaluating the model and find the best one
            # model report is of the type dictionary (type hinting)
            model_report :dict = evaluate_model(X_train = X_train,Y_train = Y_train,X_test = X_test,Y_test = Y_test,models = models, params = params)

            best_model_score = max(sorted(model_report.values())) #find the highest score

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
            # r2score = 1 - (ssres/sstotal)
            # ssres = sum((y_true - y_pred) ** 2)
            # sstotal = sum((y_true - y_true.mean()) ** 2)
            # high r2score => good fit and vice versa
            R2_score = r2_score(Y_test, predicted)
            return R2_score


        except Exception as e:
            raise CustomException(e,sys)