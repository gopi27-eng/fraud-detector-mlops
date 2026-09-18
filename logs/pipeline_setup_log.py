import sys
from pathlib import Path
from loguru import logger

def log_setup():
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_DIR / "pipeline.log"

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        LOG_FILE,
        level="INFO",
        rotation="10 MB",       # new file after 10 MB
        retention="7 days",     # keep logs for 7 days
        compression="zip",      # compress old logs
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    return logger
