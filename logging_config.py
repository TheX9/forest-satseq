import logging
from tqdm import tqdm

# Make logger multiprocessing ready
# import multiprocessing_logging

# Configure logger
logging.basicConfig(level=logging.ERROR)
root_logger = logging.getLogger()
root_logger.disabled = True
logger = None
loggers = {}

# Generate logging handler class for tqdm
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)  
            
def create_logger(logger_name, file_handler_path, tqdm = False):
    global loggers
    global logger

    # Check if logger is already created
    if loggers.get(logger_name):
        return loggers.get(logger_name)
    else:
        # create logger with
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(f'{file_handler_path}/{logger_name}.log')
        fh.setLevel(logging.DEBUG)

        # create console handler
        if tqdm:
            ch = TqdmLoggingHandler()
        else:
            ch = logging.StreamHandler()
        
        # set higher log level 
        ch.setLevel(logging.ERROR)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.propagate = False

        loggers[logger_name] = logger

        # Enable multiprocessing support
        # multiprocessing_logging.install_mp_handler()

