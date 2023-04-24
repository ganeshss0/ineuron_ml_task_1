import sys
from src.logger import logging

class CustomException(Exception):
    
    def __init__(self, error_message, error_detail: sys) -> None:
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
    
    def __str__(self) -> str:
        return self.error_message


def error_message_detail(error: str, error_detail: sys) -> str:

    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = "Error occured in python script name [{0}] line number [{1}] error message [[2]]"

    return error_message.format(file_name, exc_tb, error)


