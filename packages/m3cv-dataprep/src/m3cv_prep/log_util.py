from logging import getLogger, Logger

# global root logger for package and registry for child loggers
root_logger = getLogger("m3cv_prep")
logger_registry = {}

def get_logger(name: str) -> Logger:
    """Get a logger for the m3cv_prep package.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if name not in logger_registry:
        logger_registry[name] = root_logger.getChild(name)
    return logger_registry[name]

def set_logging_level(level: int) -> None:
    """Set the logging level for the m3cv_prep package.

    Sets the level for the root logger and for all child loggers as
    well as for any attached handlers.

    Args:
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
    """
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)
    for logger_name in logger_registry:
        logger = logger_registry[logger_name]
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
