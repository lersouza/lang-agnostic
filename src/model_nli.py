import torch

import pytorch_lightning as pl

from dataclasses import dataclass
from data_nli import extract_seq2seq_features, DATA_DEFS, data_def_name
from datasets import load_dataset
from model_utils import get_parameter_names
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    T5ForConditionalGeneration,
    get_scheduler,
    SchedulerType,
)

from logging_utils import log_artifact


@dataclass
class Prediction:
    predicted_label: str
    expected_label: str


class NLIFinetuner(pl.LightningModule):
    """
    A module for fine tuning a `model` to the ASSIN2 Task.
    """

    def __init__(
        self,
        pretrained_model: str,
        from_flax: bool,
        use_pretraining: bool,
        learning_rate: float,
        train_dataset: str,
        train_subdataset: str,
        validation_set: str,
        batch_size: int = 32,
        max_length: int = 256,
        target_max_length: int = 5,
        xlang_dataset_name: str = None,
        xlang_subdataset_name: str = None,
        xlang_validation_set: str = None,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        adam_weight_decay: float = 0.0,
        output_dir: str = "./",
    ):
        super().__init__()

        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.data_def = DATA_DEFS[data_def_name(train_dataset, train_subdataset)]

        if xlang_dataset_name:
            self.xlang_data_def = DATA_DEFS[
                data_def_name(xlang_dataset_name, xlang_subdataset_name)
            ]

        config = AutoConfig.from_pretrained(pretrained_model)

        if use_pretraining:
            self.model = T5ForConditionalGeneration.from_pretrained(
                 pretrained_model, from_flax=from_flax, config=config
            )
        else:
            self.model = T5ForConditionalGeneration(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: torch.Tensor = None,
        **kwargs,
    ):

        """Runs the network for prediction (logits) and loss calculation."""

        if self.training:
            model_output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=target_ids,
                return_dict=True,
            )

            return model_output
        else:
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.hparams.target_max_length,
                do_sample=False,
            )

    def get_num_train_steps(self):
        batches = len(self.train_dataloader())
        effective_accum = self.trainer.accumulate_grad_batches

        return (batches // effective_accum) * self.trainer.max_epochs

    def get_optimizer_parameters(self):
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if n in decay_parameters
                ],
                "weight_decay": self.hparams.adam_weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]

        return optimizer_grouped_parameters

    def configure_optimizers(self):
        total_optimization_steps = self.get_num_train_steps()
        optimization_parameters = self.get_optimizer_parameters()

        print("Total Optimization Steps:", total_optimization_steps)

        optimizer = torch.optim.AdamW(
            optimization_parameters,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            eps=self.hparams.adam_eps,
        )

        scheduler = get_scheduler(
            SchedulerType.LINEAR,
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_optimization_steps,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        """Performs the training step, logging the loss."""
        output = self(**batch)

        self.log("train/loss", output.loss)

        return output.loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        output = self(input_ids=input_ids, attention_mask=attention_mask)

        texts = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        expected_labels = batch["label"]

        batch_size = batch["input_ids"].shape[0]
        predictions_size = len(texts)

        assert batch_size == predictions_size

        predictions, labels = [], []

        for i in range(batch_size):
            predictions.append(Prediction(texts[i], expected_labels[i]))

            labels.append(str(expected_labels[i].item()))

        # Return predictions
        return {"preds": texts, "labels": labels, "valid_outputs": predictions}

    def validation_epoch_end(self, outs):
        if self.hparams.xlang_dataset_name:
            # In the case that a cross language validation dataset
            # is specified, we calculate the results for both datasets
            self._run_validation_end(outs[0], 0)
            self._run_validation_end(outs[1], 1, "cross_")
        else:
            self._run_validation_end(outs, 0)

    def _run_validation_end(self, outs, dataloader_idx, prefix="main"):
        valid_outputs = sum([o["valid_outputs"] for o in outs], [])

        valid_predictions = sum([o["preds"] for o in outs], [])
        expected_labels = sum([o["labels"] for o in outs], [])

        right_count = [
            int(pred.strip() == exp.strip())
            for pred, exp in zip(valid_predictions, expected_labels)
        ]
        accuracy = sum(right_count) / len(valid_predictions)

        self.log(f"val/{prefix}/accuracy", accuracy, prog_bar=True)
        self.log(f"val/{prefix}/examples", len(valid_predictions))

        log_artifact(
            Prediction,
            valid_outputs,
            self.logger,
            self.current_epoch,
            self.global_step,
            f"val/{prefix}/predictions",
            self.hparams.output_dir,
        )

    def setup(self, stage):
        dataset = load_dataset(
            self.hparams.train_dataset, self.hparams.train_subdataset
        )

        features = extract_seq2seq_features(
            self.data_def,
            self.tokenizer,
            self.hparams.max_length,
            self.hparams.target_max_length,
            dataset,
        )

        self.train_dataset = features["train"]
        self.valid_dataset = features[self.hparams.validation_set]

        if self.hparams.xlang_dataset_name:
            xlang_dataset = load_dataset(
                self.hparams.xlang_dataset_name, self.hparams.xlang_subdataset_name
            )

            xlang_features = extract_seq2seq_features(
                self.xlang_data_def,
                self.tokenizer,
                self.hparams.max_length,
                self.hparams.target_max_length,
                xlang_dataset,
            )

            self.cross_valid_dataset = xlang_features[self.hparams.xlang_validation_set]

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, True)

    def val_dataloader(self):
        val_loader1 = self._create_dataloader(self.valid_dataset)

        if not self.hparams.xlang_dataset_name:
            return val_loader1

        val_loader2 = self._create_dataloader(self.cross_valid_dataset)

        return [val_loader1, val_loader2]

    def _create_dataloader(
        self,
        dataset,
        shuffle: bool = False,
        batch_size: int = None,
    ):
        return DataLoader(
            dataset,
            batch_size=batch_size or self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=4,
        )
