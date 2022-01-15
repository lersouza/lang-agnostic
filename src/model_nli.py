from nis import match
import pytorch_lightning as pl
import re
import torch

from datasets import load_metric
from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration


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
            self.model = T5ForConditionalGeneration.from_pretrained(
                pretrained_model_name, from_flax=from_flax, config=config
            )
        else:
            self.model = T5ForConditionalGeneration(config)

        self.validation_metric = load_metric(metric_name)

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
                max_length=self.hparams.max_target_length,
                do_sample=False,
            )

    def training_step(self, batch, batch_idx):
        """Performs the training step, logging the loss."""
        output = self(**batch)

        self.log("train/loss", output.loss)
        self.log("train/seq_len", batch["input_ids"].shape[1])

        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)

        predictions = self._convert_to_numeric_label(output)
        references = batch["labels"]

        self.validation_metric.add_batch(predictions=predictions, references=references)

        self.log("val/seq_len", batch["input_ids"].shape[1])

    def validation_epoch_end(self, outs):
        metrics = self.validation_metric.compute()

        for metric in metrics.keys():
            self.log(f"val/{metric}", torch.tensor(metrics[metric]), prog_bar=True)

    def _convert_to_numeric_label(self, predicted_values: torch.Tensor) -> torch.Tensor:
        s2i = lambda text: int(text.strip()) if re.match("^\d+$", text.strip()) else -1

        prediction_texts = self.tokenizer.batch_decode(
            predicted_values, skip_special_tokens=True
        )
        predicted_labels = [s2i(t) for t in prediction_texts]

        return predicted_labels
