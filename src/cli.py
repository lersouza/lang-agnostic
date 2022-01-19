import os

from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI


class CustomCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
