"""
Logging utilities for the federated learning project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import datetime


def setup_logger(
    name: str = "federated_learning",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_federated_logger(
    client_id: Optional[str] = None,
    log_dir: str = "logs",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logger for federated learning with client-specific naming.
    
    Args:
        client_id: Client ID for naming
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if client_id is not None:
        logger_name = f"fl_client_{client_id}"
        log_file = f"{log_dir}/client_{client_id}_{timestamp}.log"
    else:
        logger_name = "fl_server"
        log_file = f"{log_dir}/server_{timestamp}.log"
    
    # Custom format for federated learning
    format_string = '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
    
    return setup_logger(
        name=logger_name,
        level=level,
        log_file=log_file,
        console_output=True,
        format_string=format_string
    )


def log_experiment_info(
    logger: logging.Logger,
    config: dict,
    model_info: Optional[dict] = None
):
    """
    Log experiment configuration and model information.
    
    Args:
        logger: Logger instance
        config: Experiment configuration
        model_info: Model information dictionary
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 60)
    
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"  {sub_key}: {sub_value}")
        else:
            logger.info(f"{key}: {value}")
    
    if model_info:
        logger.info("=" * 60)
        logger.info("MODEL INFORMATION")
        logger.info("=" * 60)
        
        for key, value in model_info.items():
            logger.info(f"{key}: {value}")
    
    logger.info("=" * 60)


def log_training_metrics(
    logger: logging.Logger,
    epoch: int,
    metrics: dict,
    prefix: str = "Train"
):
    """
    Log training metrics in a consistent format.
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        metrics: Dictionary of metrics
        prefix: Prefix for log messages
    """
    metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"{prefix} Epoch {epoch:3d} | {metric_str}")


def log_federated_round(
    logger: logging.Logger,
    round_num: int,
    num_clients: int,
    metrics: dict
):
    """
    Log federated learning round information.
    
    Args:
        logger: Logger instance
        round_num: Current round number
        num_clients: Number of participating clients
        metrics: Aggregated metrics
    """
    logger.info("=" * 50)
    logger.info(f"FEDERATED ROUND {round_num}")
    logger.info("=" * 50)
    logger.info(f"Participating clients: {num_clients}")
    
    if metrics:
        logger.info("Aggregated metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    logger.info("=" * 50)


class LoggerContext:
    """Context manager for temporary logger configuration."""
    
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.new_level = level
        self.old_level = logger.level
    
    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


def suppress_logging(logger: logging.Logger):
    """
    Context manager to temporarily suppress logging.
    
    Args:
        logger: Logger to suppress
        
    Returns:
        Context manager
    """
    return LoggerContext(logger, logging.CRITICAL + 1) 