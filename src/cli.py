import os

from pytorch_lightning.utilities.cli import LightningCLI


class CustomCLI(LightningCLI):
    neptune_credentials_file = "neptune_credentials.txt"

    def before_instantiate_classes(self):
        self._ensure_dirs()
        self._ensure_neptune_credentials()

    def _ensure_dirs(self):
        subcommand = self.config["subcommand"]

        os.makedirs(self.config[subcommand]["model"]["output_dir"], exist_ok=True)
        os.makedirs(
            self.config[subcommand]["trainer"]["default_root_dir"], exist_ok=True
        )

    def _ensure_neptune_credentials(self):
        if not os.path.exists(CustomCLI.neptune_credentials_file):
            project_name = input("Please, provide a Neptune project name: ")
            api_key = input("Please, provide a Neptune API Key: ")

            with open(
                CustomCLI.neptune_credentials_file, "w+", encoding="utf-8"
            ) as creds:
                creds.write("\n".join([project_name, api_key]))

        else:
            with open(
                CustomCLI.neptune_credentials_file, "r", encoding="utf-8"
            ) as creds:
                project_name, api_key = [line.strip() for line in creds.readlines()]

        os.environ["NEPTUNE_PROJECT"] = project_name
        os.environ["NEPTUNE_API_TOKEN"] = api_key
