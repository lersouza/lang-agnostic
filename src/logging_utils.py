from typing import List
from pytorch_lightning.loggers import LightningLoggerBase


def log_text(logger: LightningLoggerBase, key: str, columns: List[str], values: List[List[str]]):
    if not logger:
        return

    logger.log_text(key=key, columns=columns, data=values)
