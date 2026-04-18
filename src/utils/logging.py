"""
Logging Utilities

Centralized logging configuration for VERITAS.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional specific log file path
        log_dir: Directory for log files
        
    Returns:
        Root logger
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Generate log filename if not specified
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"veritas_{timestamp}.log"
    else:
        log_file = Path(log_file)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)
    
    # Log startup
    root_logger.info(f"Logging initialized. Level: {level}, File: {log_file}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ProofLogger:
    """
    Specialized logger for proof attempts.
    
    Tracks and logs proof search progress.
    """
    
    def __init__(self, theorem: str, log_dir: str = "logs/proofs"):
        """
        Initialize proof logger.
        
        Args:
            theorem: Theorem being proved
            log_dir: Directory for proof logs
        """
        self.theorem = theorem
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create theorem-specific log
        theorem_hash = hash(theorem) % 10000
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"proof_{theorem_hash}_{timestamp}.log"
        
        self.logger = logging.getLogger(f"proof_{theorem_hash}")
        self.logger.setLevel(logging.DEBUG)
        
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        self.logger.addHandler(handler)
        
        # Log theorem
        self.logger.info(f"Theorem: {theorem}")
        self.logger.info("-" * 50)
        
        # Track statistics
        self.step_count = 0
        self.valid_steps = 0
        self.invalid_steps = 0
    
    def log_step(
        self,
        step: str,
        is_valid: bool,
        value: float,
        signals: dict,
    ):
        """Log a proof step attempt."""
        self.step_count += 1
        
        if is_valid:
            self.valid_steps += 1
            status = "✓"
        else:
            self.invalid_steps += 1
            status = "✗"
        
        self.logger.info(
            f"Step {self.step_count} {status}: {step} "
            f"(value={value:.3f}, signals={signals})"
        )
    
    def log_expansion(
        self,
        node_id: str,
        tactics: list,
    ):
        """Log MCTS expansion."""
        self.logger.debug(f"Expanding {node_id} with {len(tactics)} tactics")
    
    def log_backprop(
        self,
        path_length: int,
        value: float,
    ):
        """Log backpropagation."""
        self.logger.debug(f"Backprop: path_length={path_length}, value={value:.3f}")
    
    def log_reflection(
        self,
        reflection_type: str,
        analysis: str,
    ):
        """Log reflection result."""
        self.logger.info(f"Reflection ({reflection_type}): {analysis}")
    
    def log_success(self, proof_steps: list):
        """Log successful proof."""
        self.logger.info("=" * 50)
        self.logger.info("PROOF FOUND")
        self.logger.info("=" * 50)
        for i, step in enumerate(proof_steps, 1):
            self.logger.info(f"  {i}. {step}")
        self.logger.info("=" * 50)
        self._log_summary()
    
    def log_failure(self, best_path: list):
        """Log failed proof attempt."""
        self.logger.info("=" * 50)
        self.logger.info("PROOF NOT FOUND")
        self.logger.info(f"Best partial path: {best_path}")
        self.logger.info("=" * 50)
        self._log_summary()
    
    def _log_summary(self):
        """Log attempt summary."""
        self.logger.info(f"Total steps tried: {self.step_count}")
        self.logger.info(f"Valid steps: {self.valid_steps}")
        self.logger.info(f"Invalid steps: {self.invalid_steps}")
        self.logger.info(
            f"Success rate: {self.valid_steps / max(1, self.step_count):.2%}"
        )
