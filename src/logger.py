import logging 
"""
The logging module in Python is used to record and manage log messages.
It provides a flexible and configurable way to track the events happening in your application,
making it easier to debug, monitor, and analyze your code's behavior.
"""
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" #strftime allows us to format a datetime object into human readable string representation.
log_path = os.path.join(os.getcwd(),"logs",LOG_FILE)
"""
os.getcwd(): This function returns the current working directory (the directory where the Python script is currently running).
"logs": This is a string representing the name of a subdirectory within the current working directory.
"""
os.makedirs(log_path,exist_ok=True) 
"""
exist_ok=True: This optional argument tells os.makedirs() to not raise an error if the directory already exists. 
If this argument is not set, the function will raise an FileExistsError if the directory already exists.
"""
LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,

)
"""The logging module provides a hierarchy of logging levels that determine the severity of log messages.
These levels range from DEBUG (least severe) to CRITICAL (most severe).
Messages at a given level and all higher levels will be logged."""