'''
Utils holds the pieces of codes that can be used in muliple places of the project
(it is ensentially the common functions that are required in the project)

'''

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import CustomException
import dill # dill is used to serialize and save an object


def save_object(file_path, obj): # it is used in data transformation is save the preprocessor as a pickle file
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
    
def evaluate_model(X_train,Y_train,X_test,Y_test,models):
    try:
        report = {}
         
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train,Y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(Y_train,y_train_pred)

            test_model_score = r2_score(Y_test,y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)
        