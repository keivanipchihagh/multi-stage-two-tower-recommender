import time

# Third-party
from src.utils import logger

def log_execution_time(func):

    def wrapper(self, *args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(self, *args, **kwargs)
        end_time = time.time()  # Record end time
        duration = end_time - start_time  # Calculate the duration
        logger.info(f"`{self.__class__.__name__}.{func.__name__}()` executed in {duration:.1f} seconds.")
        return result

    return wrapper
