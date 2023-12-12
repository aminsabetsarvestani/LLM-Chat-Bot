# app/logging_config.py

import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logger():
    # Create a custom logger
    logger = logging.getLogger("llm_as_a_service_logger")
    logger.setLevel(logging.DEBUG)  # you can set this to be more restrictive

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = RotatingFileHandler('llm_as_a_service_app.log', maxBytes=5000000, backupCount=5)  # logs rotation
    c_handler.setLevel(logging.WARNING)  # stdout handler can have a different level
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger
