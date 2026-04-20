import logging
from logging.config import dictConfig
from pathlib import Path
import sys

# Resolve log path relative to this package so cwd does not matter
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "app.log"

# Define the logging configuration
log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "default",
            "filename": str(_LOG_FILE),
            "encoding": "utf-8",
            "maxBytes": 10485760,
            "backupCount": 5,
        },
    },
    "loggers": {
        "app": {"handlers": ["console", "file"], "level": "DEBUG", "propagate": False},
    },
    "root": {"handlers": ["console", "file"], "level": "DEBUG"},
}

# Apply the configuration
dictConfig(log_config)

# Create a global logger instance
logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)
