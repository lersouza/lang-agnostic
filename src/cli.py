import os

from pytorch_lightning.utilities.cli import LightningCLI


class CustomCLI(LightningCLI):
    trainer_params_of_interest = [
        "accumulate_grad_batches",
        "gradient_clip_val",
        "max_epochs",
        "max_steps",
        "precision",
    ]

    def before_instantiate_classes(self) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"

    def before_fit(self):
        self._log_additional_hyperparameters("fit")

    def _log_additional_hyperparameters(self, subcommand):
        if not self.trainer.logger:
            return

        additional_hparams = {}

        additional_hparams.update(self._get_optimizer_params(subcommand))
        additional_hparams.update(self._get_trainer_params(subcommand))

        self.trainer.logger.log_hyperparams(additional_hparams)

    def _get_trainer_params(self, subcommand):
        trainer_config = self.config[subcommand]["trainer"]
        trainer_params = {k: trainer_config[k] for k in self.trainer_params_of_interest}

        return trainer_params

    def _get_optimizer_params(self, subcommand):
        optim_config = self.config[subcommand]["optimizer"]
        optimizer_name = optim_config["class_path"].split(".")[-1]

        return {"optimizer": optimizer_name, **optim_config["init_args"]}
