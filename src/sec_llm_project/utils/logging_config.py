#!/usr/bin/env python3
"""Centralized logging configuration for the entire project."""

import logging
import os
import sys
from pathlib import Path


class ProjectLogger:
    """Centralized logger for the project with consistent formatting and handlers."""

    _initialized = False
    _log_dir = "logs"
    _log_file = "project.log"

    @classmethod
    def setup(
        cls,
        name: str = "sec_llm_project",
        level: str = "INFO",
        console: bool = True,
        file_logging: bool = True,
        log_dir: str | None = None,
    ) -> logging.Logger:
        """
        Set up project-wide logging configuration.

        Args:
            name: Logger name (usually module name)
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            console: Enable console output
            file_logging: Enable file logging
            log_dir: Custom log directory (default: logs/)

        Returns:
            Configured logger instance
        """
        if log_dir:
            cls._log_dir = log_dir

        # Create log directory
        if file_logging:
            os.makedirs(cls._log_dir, exist_ok=True)

        # Configure root logger only once
        if not cls._initialized:
            cls._configure_root_logger(level, console, file_logging)
            cls._initialized = True

        # Return named logger
        return logging.getLogger(name)

    @classmethod
    def _configure_root_logger(cls, level: str, console: bool, file_logging: bool):
        """Configure the root logger with handlers and formatters."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper()))

        # Clear existing handlers to avoid duplicates
        root_logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # File handler
        if file_logging:
            log_path = Path(cls._log_dir) / cls._log_file
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        # Reduce noise from third-party libraries by default
        noisy_lib_loggers = [
            "httpx",
            "httpxthrottlecache",
            "edgar",
            "urllib3",
            "requests",
        ]
        for lib_name in noisy_lib_loggers:
            logging.getLogger(lib_name).setLevel(logging.WARNING)

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger instance, auto-initializing with defaults if needed."""
        if not cls._initialized:
            cls.setup()
        return logging.getLogger(name)


def setup_logging(
    verbose: bool = False,
    module_name: str = "sec_llm_project",
    console: bool = True,
    file_logging: bool = True,
) -> logging.Logger:
    """
    Convenience function to set up logging with common defaults.

    Args:
        verbose: Enable DEBUG level logging
        module_name: Name for the logger
        console: Enable console output
        file_logging: Enable file logging

    Returns:
        Configured logger instance
    """
    level = "DEBUG" if verbose else "INFO"
    return ProjectLogger.setup(
        name=module_name, level=level, console=console, file_logging=file_logging
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return ProjectLogger.get_logger(name)
