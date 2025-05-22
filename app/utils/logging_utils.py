import logging
import sys
import os
from app.config import settings

def setup_logging():
    """
    Configures the root logger for the application.
    Logs to console and optionally to a file based on settings.
    """
    log_level_name = settings.LOG_LEVEL.upper()
    log_level = getattr(logging, log_level_name, logging.INFO) # Default to INFO if invalid
    
    # Getting the root logger. Specific module loggers can be created using logging.getLogger(__name__)
    logger = logging.getLogger() # Get root logger
    logger.setLevel(log_level)

    # Prevent duplicate handlers if called multiple times (e.g., in tests or reloads)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(settings.LOG_FORMAT)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Optional)
    if settings.LOG_FILE_PATH:
        try:
            log_dir = os.path.dirname(settings.LOG_FILE_PATH)
            if log_dir and not os.path.exists(log_dir): # Ensure directory exists
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(settings.LOG_FILE_PATH, mode='a') # Append mode
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            # Use the root logger to log this message, as module-level loggers might not be configured yet
            logging.info(f"Logging to file: {settings.LOG_FILE_PATH}")
        except Exception as e:
            logging.error(f"Failed to configure file logging to {settings.LOG_FILE_PATH}: {e}", exc_info=True)

    logging.info(f"Logging configured. Level: {log_level_name}. Console: True. File: {settings.LOG_FILE_PATH or 'Disabled'}")

# Call setup_logging() once when the application starts, 
# e.g., at the beginning of main.py or a config loading module.
