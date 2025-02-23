import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    """
    This class is responsible for making predictions using the trained model.
    It loads the pre-trained model and preprocessor, transforms the input data,
    and returns the predicted results.
    """
    def __init__(self):
        pass  # No initialization required for now

    def predict(self, features):
        """
        Predicts the output based on the provided features.

        Parameters:
        features (pd.DataFrame): Input data containing feature values.

        Returns:
        np.array: Predicted values
        """
        try:
            # Define paths for the trained model and preprocessor
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            print("Before Loading")
            
            # Load the trained model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            print("After loading")
            
            # Transform the input data using the preprocessor (basically the transformations done in the data transformation file (preprocessor.pkl) are applied onto the features)
            data_scaled = preprocessor.transform(features)
            
            # Make predictions using the loaded model 
            preds = model.predict(data_scaled)
            
            return preds
        except Exception as e:
            # Raise a custom exception in case of an error
            raise CustomException(e, sys)

class CustomData:
    """
    This class is responsible for mapping the input values from the frontend to the backend.
    It structures the user-provided input into a format that the model can process.
    """
    def __init__(self, 
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        """
        Initializes the CustomData object with user-provided input values.

        Parameters:
        gender (str): Gender of the student
        race_ethnicity (str): Ethnic background of the student
        parental_level_of_education (str): Highest education level of parents
        lunch (str): Type of lunch program (e.g., standard, free/reduced)
        test_preparation_course (str): Whether test preparation was completed
        reading_score (int): Student's reading score
        writing_score (int): Student's writing score
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_frame(self):
        """
        Converts the input data into a pandas DataFrame.
        This formatted DataFrame is required for model prediction.

        Returns:
        pd.DataFrame: Data in a structured tabular format.
        """
        try:
            # Creating a dictionary to store input values
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }

            # Convert the dictionary into a Pandas DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)
