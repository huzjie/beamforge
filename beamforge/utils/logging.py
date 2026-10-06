"""Tiny logging helper."""
import logging
import sys


def get_logger(name="beamforge", level=None):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
    logger.setLevel(level or logging.INFO)
    logger.propagate = False
    return logger
