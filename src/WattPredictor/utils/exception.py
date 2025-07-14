import sys
import traceback

class CustomException(Exception):
    def __init__(self, message, error_detail: sys):
        try:
            _, _, exc_tb = error_detail.exc_info()
            if exc_tb is not None:
                file_name = exc_tb.tb_frame.f_code.co_filename
                line_number = exc_tb.tb_lineno
                self.message = f"Exception in {file_name}, line {line_number}: {message}"
            else:
                self.message = f"Exception: {message}" 
        except Exception:
            self.message = f"Exception (unknown location): {message}"

        super().__init__(self.message)
    
    def __str__(self):
        return self.message
