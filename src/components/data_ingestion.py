import os
import sys
from src.exception import CustomException  # Custom exception handling module
from src.logger import logging  # Logging module for tracking execution flow
import pandas as pd  # Library for data manipulation
from sklearn.model_selection import train_test_split  # Splitting dataset into train and test sets
from dataclasses import dataclass  # Used for creating data classes
from src.components.data_transformation import DataTransforamtion, DataTransformationConfig  # Importing data transformation components
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig  # Importing model trainer components

@dataclass  # A decorator that automatically generates __init__, __repr__, and other methods
# Use dataclass only if the class is meant to define variables, otherwise, use __init__
class DataIngestionConfig:
    # This class is used to specify file paths for storing raw, train, and test data
    train_data_path: str = os.path.join('artifacts', 'train.csv')  # Path for the training dataset
    test_data_path: str = os.path.join('artifacts', 'test.csv')  # Path for the test dataset
    raw_data_path: str = os.path.join('artifacts', 'data.csv')  # Path for the raw dataset
    # The 'artifacts' folder is used to store processed data files

class DataIngestion:
    def __init__(self):  # Constructor method, initializes the DataIngestion class
        self.ingestion_config = DataIngestionConfig()  # Creating an instance of DataIngestionConfig to access file paths
    
    def InitiateDataIngestion(self):
        """
        This method is responsible for reading the dataset, creating necessary directories,
        saving the raw dataset, splitting it into train and test sets, and saving them separately.
        """
        logging.info("Entered the data ingestion method or component")  # Logging the start of data ingestion
        try:
            # Reading the dataset from the specified location
            df = pd.read_csv("src/notebook/data/stud.csv")  # Dataset can be loaded from various sources like databases (MongoDB, SQL, etc.)
            logging.info('Read the dataset as a dataframe')  # Logging dataset read confirmation
            
            # Creating the directory structure for storing processed data if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            '''
            Breakdown of the above line:
            1) self.ingestion_config.train_data_path - Gets the file path for train.csv.
            2) os.path.dirname() - Extracts the directory path (i.e., 'artifacts').
            3) os.makedirs() - Creates the directory if it does not exist.
            4) exist_ok=True - Prevents errors if the directory already exists.
            '''
            
            # Saving the raw dataset into the artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)  # Ensures column names are preserved
            logging.info("Train Test Split initiated")  # Logging the split initiation
            
            # Splitting the dataset into training (80%) and testing (20%) sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Saving train and test sets as CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed.")  # Logging completion of data ingestion
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )  # Returning file paths of train and test datasets

        except Exception as e:
            raise CustomException(e, sys)  # Raising custom exception in case of an error

# Entry point of the script
if __name__ == "__main__":
    obj = DataIngestion()  # Creating an instance of DataIngestion
    train_data_path, test_data_path = obj.InitiateDataIngestion()  # Initiating data ingestion
    
    # Data Transformation Process
    data_transformation = DataTransforamtion()  # Creating an instance of DataTransformation class
    train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path, test_data_path)  # Transforming data
    
    # Model Training Process
    modeltrainer = ModelTrainer()  # Creating an instance of ModelTrainer class
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))  # Initiating model training and printing the results
