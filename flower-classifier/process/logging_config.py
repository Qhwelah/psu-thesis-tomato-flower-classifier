import logging
import os, sys

LOG_FILE_ROOT = os.path.dirname(__file__)

def setup_logging(log_file: str|None = None) -> logging.Logger:

    # Define a default log name if not specified by caller.
    if not log_file:
        log_file = os.path.join(LOG_FILE_ROOT, "app.log")
    else:
        log_file = os.path.join(LOG_FILE_ROOT, log_file)


    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configure logger
    ## Logging to specified log_file
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='w'
    )

    ## Logging to console
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(console_handler)

    ## Final config
    logger = logging.getLogger(__name__)
    logger.info(f"Writing log to the following location: '{log_file}'")
    return logger