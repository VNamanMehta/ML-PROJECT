from src.logger import logging
import sys #helps manage python runtime environment (exception handling)

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info() #gives execution info (3 infos [first 2 arent needed , 3rd is needed])
    #exc_tb giveds the details of where the error (line etc) occurs

    file_name = exc_tb.tb_frame.f_code.co_filename #from custom exception handling in python documentation

    error_message = "Error occured in python script name[{0}] line number [{1}] error message[{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message

class CustomException(Exception): #we inherit builtin exception class
    def __init__(self,error_message,error_detail:sys): #init constructor is called
        super().__init__(error_message) #This line calls the __init__ method of the parent class (Exception) with the error_message argument. This ensures that the CustomException object inherits the basic behavior of Exception
        self.error_message = error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message