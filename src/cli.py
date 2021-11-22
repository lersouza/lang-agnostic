import os

from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI

# 
class CustomCLI(LightningCLI):

    def before_instantiate_classes(self):
        self._ensure_dirs()

    def _ensure_dirs(self):
        subcommand = self.config["subcommand"]

        os.makedirs(self.config[subcommand]["model"]["output_dir"], exist_ok=True)
        os.makedirs(
            self.config[subcommand]["trainer"]["default_root_dir"], exist_ok=True
        )
