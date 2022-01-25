from collections import defaultdict
import re

import pytorch_lightning as pl
import torch

from datasets import load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM


def format_results_as_table(predictions, raw_predictions, references):
    return ["predictions", "raw_predictions", "references"], [
        [str(p), str(o), str(r)]
        for p, o, r in zip(predictions, raw_predictions, references)
    ]


def aggregate_outputs(outputs):
    if not outputs:
        return None

    keys = outputs[0].keys()
    aggregated = {k: sum([o[k] for o in outputs], []) for k in keys}

    return aggregated


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

        self.validation_metric = load_metric(metric_name)

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

    def validation_step(self, batch, batch_idx):
        output = self(**batch)

        predictions, original = self._convert_to_numeric_label(output)
        references = batch["labels"].tolist()

        self.validation_metric.add_batch(predictions=predictions, references=references)
        self.log("val/seq_len", float(batch["input_ids"].shape[1]))

        return {
            "predictions": predictions,
            "raw_predictions": original,
            "references": references,
        }

    def validation_epoch_end(self, outputs):
        out = aggregate_outputs(outputs)

        metrics = self.validation_metric.compute()
        cols, samples = format_results_as_table(
            out["predictions"], out["raw_predictions"], out["references"]
        )

        for metric in metrics.keys():
            self.log(f"val/{metric}", torch.tensor(metrics[metric]), prog_bar=True)

        self.log("val/num_predictions", float(len(out["predictions"])))
        self.log("val/num_references", float(len(out["references"])))

        self.logger.log_text(key="val/classifications", columns=cols, data=samples)

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

        return predicted_labels, prediction_texts
