"""Logging utilities for the pipeline."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logging once and return the project logger.

    Args:
        level: Logging level, usually ``logging.INFO`` or ``logging.DEBUG``.

    Returns:
        Configured logger named ``vgae_delaunay_pipeline``.
    """
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger("vgae_delaunay_pipeline")
