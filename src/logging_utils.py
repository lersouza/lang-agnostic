""" A module with utility functions for logging during experiments. """
from typing import List
from pytorch_lightning.loggers import LightningLoggerBase


def log_text(logger: LightningLoggerBase, key: str, columns: List[str], values: List[List[str]]):
    """
    Logs a text to the looger, if it supports this operation.

    Parameters:

        logger (LightningLoggerBase):
            The logger to use.

        key (str):
            The key to provide to the logger to identify these texts.

        columns (List[str]):
            A list of column names, so text is logged as a table.

        value (List[List[str]]):
            A list in which each item is another list holding the values for each column.
    """
    if not logger or not hasattr(logger, "log_text"):
        return

    logger.log_text(key=key, columns=columns, data=values)
