import re

import numpy as np
import pytorch_lightning as pl
import torch

from collections import defaultdict
from typing import List

from datasets import load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

from logging_utils import log_text


def format_results_as_table(predictions, references):
    return ["predictions", "references"], [
        [str(p), str(r)] for p, r in zip(predictions, references)
    ]


def aggregate_single_output(outputs):
    keys = outputs[0].keys()
    aggregated = {k: sum([o[k] for o in outputs], []) for k in keys}

    return aggregated


def aggregate_outputs(outputs, dataloader_names):
    if isinstance(outputs[0], list):
        assert len(dataloader_names) == len(outputs)

        return {
            k: aggregate_single_output(outputs[i])
            for i, k in enumerate(dataloader_names)
        }

    return {dataloader_names[0]: aggregate_single_output(outputs)}


class TextClassificationModel(pl.LightningModule):
    """
    A module for performing Text Classification Task (Natural Language Inference).
    """

    def __init__(
        self,
        pretrained_model_name: str,
        use_pretrained_weights: bool = True,
        max_target_length: int = 5,
        metric_name: str = "accuracy",
        from_flax: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        config = AutoConfig.from_pretrained(pretrained_model_name)

        if use_pretrained_weights:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name, from_flax=from_flax, config=config
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_config(config)

        self.metric = load_metric(metric_name)

    @property
    def val_dataloader_names(self):
        if self.trainer and self.trainer.datamodule:
            return self.trainer.datamodule.val_dataloader_names

        return ["default"]

    @property
    def test_dataloader_names(self):
        if self.trainer and self.trainer.datamodule:
            return self.trainer.datamodule.test_dataloader_names

        return ["default"]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: torch.Tensor = None,
        **kwargs,
    ):
        """Runs the network for prediction (logits) and loss calculation."""

        if self.training == True:
            # Changing pad token in target labels
            # See: https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5ForConditionalGeneration
            target_labels = target_ids.masked_fill(
                target_ids == self.tokenizer.pad_token_id, -100
            )

            model_output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=target_labels,
                return_dict=True,
            )

            return model_output
        else:
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.hparams.max_target_length,
                do_sample=False,
            )

    def training_step(self, batch, batch_idx):
        """Performs the training step, logging the loss."""
        output = self(**batch)

        self.log("train/loss", output.loss)
        self.log("train/seq_len", float(batch["input_ids"].shape[1]))

        return output.loss

    def validation_step(self, *args, **kwargs):
        return self.common_eval_step("val", *args, **kwargs)

    def validation_epoch_end(self, outputs) -> None:
        return self.common_eval_epoch_end("val", self.val_dataloader_names, outputs)

    def test_step(self, *args, **kwargs):
        return self.common_eval_step("test", *args, **kwargs)

    def test_epoch_end(self, outputs) -> None:
        return self.common_eval_epoch_end("test", self.test_dataloader_names, outputs)

    def common_eval_step(
        self, prefix: str, batch, batch_idx: int = 0, dataloader_idx: int = 0
    ):
        output = self(**batch)

        predictions = self._convert_to_numeric_label(output)
        references = batch["labels"].tolist()

        self.log(f"{prefix}/seq_len", float(batch["input_ids"].shape[1]))

        return {
            "predictions": predictions,
            "references": references,
        }

    def common_eval_epoch_end(self, prefix: str, dataloader_names: List[str], outputs):
        out = aggregate_outputs(outputs, dataloader_names)
        avg_metrics = defaultdict(list)

        for name, values in out.items():
            kwargs = {
                "predictions": values["predictions"],
                "references": values["references"],
            }

            metrics = self.metric.compute(**kwargs)
            cols, samples = format_results_as_table(**kwargs)

            for metric in metrics.keys():
                metric_name = f"{prefix}/{metric}/{name}"
                metric_value = float(metrics[metric])

                self.log(metric_name, metric_value, prog_bar=False)

                # Aggregate for an average value on prog bar
                avg_metrics[f"{prefix}/avg_{metric}"].append(metric_value)

            self.log(
                f"{prefix}/num_predictions/{name}", float(len(values["predictions"]))
            )
            self.log(
                f"{prefix}/num_references/{name}", float(len(values["references"]))
            )

            log_text(
                self.logger,
                key=f"{prefix}/classifications/{name}",
                columns=cols,
                values=samples,
            )

        for metric_name, metric_value in avg_metrics.items():
            self.log(metric_name, np.average(metric_value), prog_bar=True)

    def _convert_to_numeric_label(self, predicted_values: torch.Tensor) -> torch.Tensor:
        """
        Convert model's output tokens to an integer representation of a predicted label.

        First, the `predicted_values` is decoded using the model's tokenizer.
        Then, every value is casted as integer.

        If an invalid integer label is produced (a string "invalid", for instance),
        we assume `-1` as the output, since this corresponds to an invalid label.
        """
        s2i = lambda text: int(text.strip()) if re.match(r"^\d+$", text.strip()) else -1

        prediction_texts = self.tokenizer.batch_decode(
            predicted_values, skip_special_tokens=True
        )
        predicted_labels = [s2i(t) for t in prediction_texts]

        return predicted_labels
