import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass #Decorator - helps to directly define class variables without init
# use dataclass only if the class is ment to define variables else use init
class DataIngestionConfig: # used to give the inputs required for data ingestion such as test data path , train data path, raw file path etc
    train_data_path : str = os.path.join('artifacts','train.csv') # atrifacts - folder train.csv - file name
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','data.csv')
    #above are the inputs that allow the data ingestion component know where to save the train, test and raw data

class DataIngestion:
    def __init__(self): #self is used to access the current instance of the class
        self.ingestion_config = DataIngestionConfig() 
        # here in the ingestion_config variable the paths given in the DataIngestionClass as stored

    def InitiateDataIngestion(self):
        # use this function to read the dataset
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("src/notebook/data/stud.csv") #this reading can be done using mongodb,sql etc
            logging.info('Read the dataset as a dataframe') 
            # logging is used to deocument information. In case exception occurs logs can be accesed to understand the exception

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            '''The code above does:
                1) self.ingestion_config.train_data_path - selects the train_data_path variable from the ingestion_config variable 
                2) os.path.dirname() - selects only the dirname from the train_data_path
                3) os.makedirs() - makes a directory with the given name
                4) exists_ok = True - if the directory already exists then no new directory is created
            '''
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) # take the dataset, converts it to csv(if not in csv already) then stores it in the raw_data_path. Header=True ensures that the column names are kept
            logging.info("Train Test Split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set = train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed.")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                
            )

        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.InitiateDataIngestion()
