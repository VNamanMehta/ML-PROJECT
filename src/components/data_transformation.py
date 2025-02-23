# main purpose of data transformation is feature engineering, data cleaning, convert categorical features to numerical feratures etc
import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer  # used to handle missing values
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig: # used to give inputs required for the data transformation
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl") #pickle file is stored (created) in aritfacts directory
    # preprocessor obj is required to allow for encoding, scaling imputing etc and seamless integretion into pipelines

class DataTransforamtion:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self): # helps create pickle file for encoding , scaling etc (basically data transformation)
        try:
            numerical_features = ["writing_score","reading_score"] # input numerical columns
            categorical_features = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"] # input categorical columns

            #Pipeline is used to streamline the workflow and organize the execution of various steps in a more generalized manner

            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")), # imputer is used to handle missing values by using the median
                    ("scaler",StandardScaler()) # scaler is used to scale(bring) the values to a certain range
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")), # most_frequent - mode
                    ("one_hot_encoder",OneHotEncoder()), # converting categories into a series of binary numbers
                    ("scaler",StandardScaler(with_mean=False)) # we use with_mean = False to not effect the output of one hot encoder (as it gives 0 and 1 but scaler find the mean hence data might be altered)
                ]
            )

            logging.info(f"Numerical columns: {numerical_features}")
            logging.info(f"Categorical columns: {categorical_features}")

            # we use the column transformer to combine both the numerical and categorical pipelines

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_features), # we give a name, pipeline, columns on which it should be applied
                    ("cat_pipeline",cat_pipeline,categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
            
    
    def initiate_data_transformation(self,trian_path,test_path): # used to call the preprocessor object

        try:
            train_df = pd.read_csv(trian_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "math_score" 
            numerical_features = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1) # removing the target column from the input features
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df) # here be fit and transform the data on the preprocessor_obj (get_data_transformer_object is called on the data) 
            # It is converted into a sparse matrix - A sparse matrix is a matrix in which most of the elements are zero. Instead of storing all elements, we store only non-zero values and their positions.
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # fit_transform() on training data learns the transformation parameters.
            # transform() on test data applies the learned transformation without introducing bias.
            # This separation guarantees a fair evaluation of the model's performance on truly unseen data.

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)] # np.c_ combine the input_feature_train_arr and the target_feature_train_df into a 2D array format
            # the input feature train arr is returned by the preprocessor_obj in the form of a numpy array (because of one hot encoding else it may be a sparse matrix of a pd dataframe)
            # the target feature train df is converted into numpy format using np.array
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object( # save object method imported from the utils is called to save the pickle file

                file_path = self.data_transformation_config.preprocessor_obj_file_path, # file path - where to save 
                obj = preprocessor_obj # object - what to save
            )

            logging.info(f"Saved preprocessing object")

            return (
                train_arr,
                test_arr,
            )
        except Exception as e:
            raise CustomException(e, sys)