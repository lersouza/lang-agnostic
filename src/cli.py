import os

from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI

# 
class CustomCLI(LightningCLI):

    def before_instantiate_classes(self):
        self._ensure_dirs()

    def _ensure_dirs(self):
        subcommand = self.config["subcommand"]
        output_dir = self.config[subcommand]["model"]["output_dir"]
        logs_dir = os.path.join(output_dir, "logs/")
        checkpoint_dir = os.path.join(output_dir, "checkpoints/")

        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

