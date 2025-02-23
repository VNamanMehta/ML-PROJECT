'''
Utils holds the pieces of codes that can be used in muliple places of the project
(it is ensentially the common functions that are required in the project)

'''

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
import dill # dill is used to serialize and save an object


def save_object(file_path, obj): # it is used in data transformation and model trainer to save the preprocessor and the model as a pickle file
    try:
        dir_path = os.path.dirname(file_path) # takes the directory name from the file_path
        os.makedirs(dir_path, exist_ok=True) # makes the directory if it already does not exist
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

            '''
            with open(file_path, 'wb') as file_obj:

This line opens a file located at file_path in binary write mode ('wb'). wb- write in binary format
The with statement ensures that the file is automatically closed even if errors occur during the process.
file_obj becomes a file object representing the opened file, allowing you to interact with it.
dill.dump(obj, file_obj):

dill.dump() is the core function for serialization. It takes two main arguments:
obj: object to be saved (preprocessor)
file_obj: The file object (obtained from open()) where the serialized data will be written
            '''

    except Exception as e:
        raise CustomException(e, sys)
    
    
# Import necessary libraries
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning
from sklearn.metrics import r2_score  # For model performance evaluation
import sys  # For exception handling

def evaluate_model(X_train, Y_train, X_test, Y_test, models, params):
    """
    Evaluates multiple machine learning models using GridSearchCV for hyperparameter tuning.
    
    Args:
        X_train (array-like): Training feature set.
        Y_train (array-like): Training target values.
        X_test (array-like): Test feature set.
        Y_test (array-like): Test target values.
        models (dict): Dictionary containing different machine learning models.
        params (dict): Dictionary containing hyperparameter grids for each model.

    Returns:
        dict: A dictionary containing model names as keys and their corresponding R² scores as values.
    """
    try:
        report = {}  # Dictionary to store model evaluation scores

        # Loop through each model in the dictionary
        for model_name, model in models.items():  
            # Retrieve the corresponding hyperparameter grid for the current model
            param_grid = params.get(model_name, {})  # Get parameters; default to empty dict if not found

            '''
            instead of items() we can also use:
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            but using items() is more clean and readable
            items() returns key-value pairs (model name and model object) from the dictionary
            param = params[model_name]  # Get hyperparameters for the current model
            '''

            # Perform Grid Search Cross Validation to find the best hyperparameters
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)  
            # cv=3 means 3-fold cross-validation (data is split into 3 parts which is later trained individually, best one is chosen)
            # scoring='r2' ensures the best parameters are chosen based on R² score
            # n_jobs=-1 allows parallel computation to speed up the process

            grid_search.fit(X_train, Y_train)  # Train the model using different hyperparameter combinations

            # Retrieve the best hyperparameters found by GridSearchCV
            best_params = grid_search.best_params_  # best_params_ is a dictionary containing the best result for the model. Eg: {'n_estimators': 50, 'max_depth': 5} 

            # Set the best hyperparameters to the model
            model.set_params(**best_params)  # The ** operator unpacks the dictionary into keyword arguments. Eg: model.set_params(n_estimators=50, max_depth=5)

            # Train the model using the best hyperparameters
            model.fit(X_train, Y_train)

            # Make predictions on training and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate model performance using R² score
            train_model_score = r2_score(Y_train, y_train_pred)  
            test_model_score = r2_score(Y_test, y_test_pred)  

            # Store the test score in the report dictionary
            report[model_name] = test_model_score  

        return report  # Return the dictionary containing model performance results

    except Exception as e:
        # CustomException is a user-defined exception handler (assumed to be defined elsewhere)
        raise CustomException(e, sys)  

    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
        