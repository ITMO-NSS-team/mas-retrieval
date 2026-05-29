"""Logging utilities for SwarmAgentic adapter."""

import json
import logging
from pathlib import Path

BASE_PATH = Path("logs/swarm_agentic")


def setup_logger(name):
    """Set up a file logger for SwarmAgentic runs."""
    logger = logging.getLogger(f"swarm-{name}")
    logger.setLevel(logging.WARNING)

    BASE_PATH.mkdir(parents=True, exist_ok=True)

    log_path = BASE_PATH / f"result-{name}.log"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.WARNING)

    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


def log(logger, mode, input, output=None, mark=None):
    """Log a message with mode, input, and optional output."""
    if logger is None:
        return

    if output:
        if mode in ("Init Team", "Update Team"):
            roles = json.dumps(output["roles"], indent=2)
            workflow = json.dumps(output["workflow"], indent=2)
            output = f"# Roles:\n{roles}\n\n# Workflow:\n{workflow}"

        logger.warning(f"==========={mode} Input===========\n{input}\n")
        logger.warning(f"==========={mode} Output===========\n{output}\n")
    else:
        if mark == "-":
            logger.warning(f"-----------{mode}-----------\n{input}\n")
        else:
            logger.warning(f"==========={mode}===========\n{input}\n")


def log_all(logger, logs):
    """Log all items in logs list."""
    if logger is None:
        return
    for item in logs:
        log(logger=logger, mode=item[0], input=item[1], output=item[2])
