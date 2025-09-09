import logging


def setup_logging(name: str, console_level: str, file_level: str) -> logging.Logger:
    """
    Configure and return a logger with console and file handlers.

    Creates a logger with dual output: console (stdout) and file ("app.log").
    Both handlers use the same timestamp format but can have different log levels.

    Parameters
    ----------
    name : str
        Name for the logger, typically __name__ of the calling module.
    console_level : str
        Log level for console output (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
    file_level : str
        Log level for file output (e.g., "DEBUG", "INFO", "WARNING", "ERROR").

    Returns
    -------
    logging.Logger
        Configured logger instance with console and file handlers.

    Notes
    -----
    The logger is configured with:
    - Base level set to "DEBUG" to capture all messages
    - Timestamp format: "YYYY-MM-DD HH:MM:SS.mmm"
    - File output appended to "app.log" in UTF-8 encoding
    - Both handlers use identical formatting

    Examples
    --------
    >>> logger = setup_logging(__name__, "INFO", "DEBUG")
    >>> logger.info("This appears on console and in file")
    >>> logger.debug("This only appears in file")
    """
    logger = logging.getLogger(name)
    logger.setLevel("DEBUG")
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d' - %(levelname)s - %(message)s",
        style="%",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(file_level)
    logger.addHandler(file_handler)

    return logger
