""" A base Module for Seq2Seq Models. """

from abc import abstractmethod
from collections import defaultdict
import time
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from datasets import Dataset, DatasetDict, load_metric
from pytorch_lightning import LightningModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, BatchEncoding

from logging_utils import log_text


class BaseSeq2SeqModel(LightningModule):
    """
    A base model for using Transformer's Seq2Seq API in Pytorch Lightning.
    """

    # pylint: disable=arguments-differ

    def __init__(
        self,
        pretrained_model_name: str,
        pretrained_model_revision: str = None,
        use_pretrained_weights: bool = True,
        max_target_length: int = 128,
        metric_name: str = "accuracy",
        from_flax: bool = False,
    ):
        # pylint: disable=unused-argument

        super().__init__()

        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        config = AutoConfig.from_pretrained(pretrained_model_name)

        if use_pretrained_weights:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name,
                from_flax=from_flax,
                config=config,
                revision=pretrained_model_revision,
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_config(config)

        self.metric = load_metric(metric_name)

    @property
    def val_dataloader_names(self):
        """
        Returns the names of the validation Data Loaders.
        """
        if self.trainer and self.trainer.datamodule:
            return self.trainer.datamodule.dataloader_names["validation"]

        return ["default"]

    @property
    def test_dataloader_names(self):
        """
        Returns the names of the test Data Loaders.
        """
        if self.trainer and self.trainer.datamodule:
            return self.trainer.datamodule.dataloader_names["test"]

        return ["default"]

    @abstractmethod
    def format_model_predictions(
        self,
        dataloader_index: int,
        dataloader_name: str,
        batch: Dict[str, Union[torch.Tensor, Any]],
        output: torch.Tensor,
    ) -> List[Any]:
        """
        Formats the model Predictions according to the chosen metric.

        Parameters:
            dataloader_index (int):
                The index of the dataloader being processed at this time

            dataloader_name (str):
                The name of the dataloader being processed at this time

            batch (Dict[str, Union[Tensor, Any]]):
                The batch for which predictions were made

            output (torch.Tensor):
                The model's computed output for the batch.
        """

    @abstractmethod
    def format_batch_references(
        self,
        dataloader_index: int,
        dataloader_name: str,
        batch: Dict[str, Union[torch.Tensor, Any]],
    ):
        """
        Formats the References in the batch for comparing against model's predictions.

        Parameters:
            dataloader_index (int):
                The index of the dataloader being processed at this time

            dataloader_name (str):
                The name of the dataloader being processed at this time

            batch (Dict[Union[Tensor, Any]]):
                The batch for which predictions were made.
        """

    def format_outputs_for_logging(self, predictions, references) -> Tuple:
        """
        Hook method to provide additional textual information for logging.
        Returns a tuble in which:

            Element[0]:
                Column names for the data to be logged

            Element[1]:
                A list of lists cotaining all examples to be logged, described
                by its columns.
        """
        # pylint: disable=unused-argument

        return ()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: torch.Tensor = None,
        **kwargs,
    ):
        """Runs the network for prediction (logits) and loss calculation."""

        if self.training is True:
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
        """Performs the default training step, logging the loss and seq length for the batch."""
        # pylint: disable=unused-argument

        output = self(**batch)

        self.log("train/loss", output.loss)
        self.log("train/seq_len", float(batch["input_ids"].shape[1]))

        return output.loss

    def validation_step(self, batch, batch_idx: int = 0, dataloader_idx: int = 0):
        return self.common_eval_step(
            "val", batch, batch_idx, dataloader_idx, self.val_dataloader_names
        )

    def test_step(self, batch, batch_idx: int = 0, dataloader_idx: int = 0):
        return self.common_eval_step(
            "test", batch, batch_idx, dataloader_idx, self.test_dataloader_names
        )

    def common_eval_step(
        self,
        prefix: str,
        batch,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
        dataloader_names: List[str] = None,
    ):
        """
        Performs an evaluation step, common to both validation and test phases.

        Parameters:

            prefix (str):
                A prefix to be appended to logs identifying the phase (i.e., val or test).

            batch (Dict[str, Any]):
                The batch to process during this step.

            batch_idx (int, optional, default = 0):
                The index of the batch in the whole dataset.

            dataloader_idx (int, optional, default = 0):
                The index of the dataloader being processed when using multiple ones.

            dataloader_names (List[str], optional. If None, `["default"]` is used):
                The names of the known dataloaders.
        """
        # pylint: disable=unused-argument

        dataloader_names = dataloader_names or ["default"]

        output = self(**batch)
        current_dataloader_name = dataloader_names[dataloader_idx]

        predictions = self.format_model_predictions(
            dataloader_idx,
            current_dataloader_name,
            batch,
            output,
        )
        references = self.format_batch_references(
            dataloader_idx,
            current_dataloader_name,
            batch,
        )

        self.log(f"{prefix}/seq_len", float(batch["input_ids"].shape[1]))

        return {
            "predictions": predictions,
            "references": references,
        }

    def validation_epoch_end(self, outputs) -> None:
        return self.common_eval_epoch_end("val", self.val_dataloader_names, outputs)

    def test_epoch_end(self, outputs) -> None:
        return self.common_eval_epoch_end("test", self.test_dataloader_names, outputs)

    def common_eval_epoch_end(self, prefix: str, dataloader_names: List[str], outputs):
        """
        Performs an evaluation final processing at the epoch end,
        common to both validation and test phases.

        Parameters:

            prefix (str):
                A prefix to be appended to logs identifying the phase (i.e., val or test).

            dataloader_names (List[str], optional. If None, `["default"]` is used):
                The names of the known dataloaders.

            outputs (List[Any]):
                A list containing the outputs produced during steps.
        """
        out = self.aggregate_outputs(outputs, dataloader_names)
        avg_metrics = defaultdict(list)

        for name, dl_values in out.items():
            self.process_dataloader_results(prefix, avg_metrics, name, dl_values)

        for metric_name, metric_value in avg_metrics.items():
            self.log(metric_name, np.average(metric_value), prog_bar=True)

    def process_dataloader_results(
        self,
        prefix: str,
        accumulated_metrics: Dict[str, List],
        dl_name: str,
        dl_values: Dict[str, List],
    ):
        """
        Process the validation/test outputs for a given Data Loader.

        Parameters:

            prefix (str):
                The prefix for metrics and logs. Could be "val" or "test", for instance.

            accumulated_metrics (Dict[str, List]):
                A dictionary for accumulated metrics. This dictionary will be updated by adding
                values for metrics with keys following the format `{prefix}/avg_{metric_name}`.

            dl_name (str):
                The name of the dataloder being processed.

            dl_values (Dict[str, List]):
                A dictionary with all outputs produced by steps for that dataloader.
        """
        kwargs = {
            "predictions": dl_values["predictions"],
            "references": dl_values["references"],
        }

        metrics = self.metric.compute(**kwargs)

        for metric in metrics.keys():
            metric_name = f"{prefix}/{metric}/{dl_name}"
            metric_value = float(metrics[metric])

            self.log(metric_name, metric_value, prog_bar=False)

            # Aggregate for an average value on prog bar
            accumulated_metrics[f"{prefix}/avg_{metric}"].append(metric_value)

        self.log(
            f"{prefix}/num_predictions/{dl_name}", float(len(dl_values["predictions"]))
        )
        self.log(
            f"{prefix}/num_references/{dl_name}", float(len(dl_values["references"]))
        )

        additional_logging = self.format_outputs_for_logging(**kwargs)

        if additional_logging:
            cols, samples = additional_logging

            log_text(
                self.logger,
                key=f"{prefix}/predictions/{dl_name}",
                columns=cols,
                values=samples,
            )

    @staticmethod
    def aggregate_single_output(outputs: List[Dict[str, List]]):
        """
        Aggregates the outputs of a Dataloader accross many batches.

        Parameters:

            outputs (List[Dict[str, List]]):
                A list of the outputs retrieved by processing each batch.
        """
        if not outputs:
            return {}

        keys = outputs[0].keys()
        aggregated = {k: sum([o[k] for o in outputs], []) for k in keys}

        return aggregated

    @staticmethod
    def aggregate_outputs(outputs, dataloader_names):
        """
        Aggregates eval outputs from multiple dataloaders (or a single one)
        into a dictionary in the following format:

        ```
        {
            data_loader_name: {
                output_item_1: [ ... ],  # list of all batches' outputs
                output_item_2: [ ... ],
            },
            ...
        }
        ```
        """
        if isinstance(outputs[0], list):
            assert len(dataloader_names) == len(outputs)

            return {
                k: BaseSeq2SeqModel.aggregate_single_output(outputs[i])
                for i, k in enumerate(dataloader_names)
            }

        return {dataloader_names[0]: BaseSeq2SeqModel.aggregate_single_output(outputs)}


class BaseSeq2SeqModelV2(LightningModule):
    """
    A base model for using Transformer's Seq2Seq API in Pytorch Lightning.
    """

    # pylint: disable=arguments-differ
    # pylint: disable=abstract-method

    def __init__(
        self,
        pretrained_model_name: str,
        metric_name: str,
        pretrained_model_revision: str = None,
        use_pretrained_weights: bool = True,
        max_target_length: int = 128,
        from_flax: bool = False,
    ):
        # pylint: disable=unused-argument

        super().__init__()

        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        config = AutoConfig.from_pretrained(pretrained_model_name)

        if use_pretrained_weights:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name,
                from_flax=from_flax,
                config=config,
                revision=pretrained_model_revision,
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_config(config)

        self.metric = load_metric(metric_name)

    @property
    def val_dataloader_names(self):
        """
        Returns the names of the validation Data Loaders.
        """
        if self.trainer and self.trainer.datamodule:
            return self.trainer.datamodule.val_dataloader_names

        return ["default"]

    @property
    def test_dataloader_names(self):
        """
        Returns the names of the test Data Loaders.
        """
        if self.trainer and self.trainer.datamodule:
            return self.trainer.datamodule.test_dataloader_names

        return ["default"]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        """Runs the network for prediction (logits) and loss calculation."""

        if self.training is True:
            # Changing pad token in target labels
            # See: https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5ForConditionalGeneration
            labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

            model_output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
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

    def _get_examples_and_features(self, split) -> DatasetDict:
        """
        Gets a tuple of examples ([0]) and features ([1])
        currently associated with this module.

        Returns a DatasetDict. If there is only one Dataset for
        examples or features, a DatasetDict with a `default` entry is returned.
        """
        assert self.trainer and self.trainer.datamodule, "Datamodule not available"
        assert hasattr(self.trainer.datamodule, "data"), "Data not found"
        assert hasattr(self.trainer.datamodule, "features"), "Features not found"

        examples = self.trainer.datamodule.data[split]
        features = self.trainer.datamodule.features[split]

        if isinstance(examples, Dataset):
            examples = DatasetDict({"default": examples})

        if isinstance(features, Dataset):
            features = DatasetDict({"default": features})

        return examples, features

    def training_step(self, batch, batch_idx):
        """Performs the default training step, logging the loss and seq length for the batch."""
        # pylint: disable=unused-argument

        output = self(**batch)

        self.log("train/loss", output.loss)
        self.log("train/seq_len", float(batch["input_ids"].shape[1]))

        return output.loss

    def validation_step(self, batch, batch_idx: int = 0, dataloader_idx: int = 0):
        return self._evaluation_step("val", batch)

    def test_step(self, batch, batch_idx: int = 0, dataloader_idx: int = 0):
        return self._evaluation_step("test", batch)

    def _evaluation_step(self, prefix: str, batch: BatchEncoding):
        """
        Performs an evaluation step, common to both validation and test phases.

        Parameters:

            prefix (str):
                A prefix to be appended to logs identifying the phase (i.e., val or test).

            batch (Dict[str, Any]):
                The batch to process during this step.

            batch_idx (int, optional, default = 0):
                The index of the batch in the whole dataset.

            dataloader_idx (int, optional, default = 0):
                The index of the dataloader being processed when using multiple ones.

            dataloader_names (List[str], optional. If None, `["default"]` is used):
                The names of the known dataloaders.
        """
        # pylint: disable=unused-argument
        batch_size, input_seq_len = batch["input_ids"].shape[:2]

        start_time = time.time()
        output = self(**batch)
        gen_time = (time.time() - start_time) / batch_size

        self.log(f"{prefix}/seq_len", float(input_seq_len))
        self.log(f"{prefix}/gen_time", gen_time)

        return {"predictions": output}

    def validation_epoch_end(self, outputs) -> None:
        examples, features = self._get_examples_and_features("validation")

        return self._evaluation_epoch_end(
            "val",
            self.val_dataloader_names,
            examples,
            features,
            outputs,
        )

    def test_epoch_end(self, outputs) -> None:
        examples, features = self._get_examples_and_features("test")

        return self._evaluation_epoch_end(
            "test",
            self.test_dataloader_names,
            examples,
            features,
            outputs,
        )

    def _evaluation_epoch_end(
        self,
        prefix: str,
        dataloader_names: List[str],
        examples: DatasetDict,
        features: DatasetDict,
        outputs,
    ):
        """
        Performs an evaluation final processing at the epoch end,
        common to both validation and test phases.

        Parameters:

            prefix (str):
                A prefix to be appended to logs identifying the phase (i.e., val or test).

            dataloader_names (List[str], optional. If None, `["default"]` is used):
                The names of the known dataloaders.

            outputs (List[Any]):
                A list containing the outputs produced during steps.
        """
        out = self.aggregate_outputs(outputs, dataloader_names)
        avg_metrics = defaultdict(list)

        for name, predictions in out.items():
            self.process_dataloader_results(
                prefix,
                avg_metrics,
                name,
                examples[name],
                features[name],
                predictions,
            )

        for metric_name, metric_value in avg_metrics.items():
            self.log(metric_name, np.average(metric_value), prog_bar=True)

    def process_dataloader_results(
        self,
        prefix: str,
        accumulated_metrics: Dict[str, List],
        dl_name: str,
        examples: Dataset,
        features: Dataset,
        model_outputs: Dict[str, List],
    ):
        """
        Process the validation/test outputs for a given Data Loader.

        Parameters:

            prefix (str):
                The prefix for metrics and logs. Could be "val" or "test", for instance.

            accumulated_metrics (Dict[str, List]):
                A dictionary for accumulated metrics. This dictionary will be updated by adding
                values for metrics with keys following the format `{prefix}/avg_{metric_name}`.

            dl_name (str):
                The name of the dataloder being processed.

            examples (Dataset):
                A dataset with the original examples used by this module.

            features (Dataset):
                A dataset with the features provided to the model for making predictions.

            model_outputs (Dict[str, List]):
                A dictionary with all outputs produced by steps for that dataloader.
        """
        result = self.post_process(examples, features, model_outputs)
        metrics = self.metric.compute(**result)

        for metric in metrics.keys():
            metric_name = f"{prefix}/{metric}/{dl_name}"
            metric_value = float(metrics[metric])

            self.log(metric_name, metric_value, prog_bar=False)

            # Aggregate for an average value on prog bar
            accumulated_metrics[f"{prefix}/avg_{metric}"].append(metric_value)

        self.log(
            f"{prefix}/num_predictions/{dl_name}", float(len(result["predictions"]))
        )
        self.log(
            f"{prefix}/num_references/{dl_name}", float(len(result["references"]))
        )

        additional_logging = self.format_outputs_for_logging(**result)

        if additional_logging:
            cols, samples = additional_logging

            log_text(
                self.logger,
                key=f"{prefix}/predictions/{dl_name}",
                columns=cols,
                values=samples,
            )

    @abstractmethod
    def post_process(self, examples: Dataset, features: Dataset, model_outputs: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Post process the results from the model and returns a dictionary with the following keys:

            predictions:
                a list of predictions made by the model.

            references:
                a list of references to compare the predictions against.

        Parameters:

            examples (Dataset):
                A dataset cotaning the original examples for which predictions were made

            features (Dataset):
                A dataset with the actual features provided for the model to make predictions.

            model_outputs (Dict[List[Any]]):
                A Dictionary with all outputs from the prediction step (eval or test)
        """

    @staticmethod
    def aggregate_single_output(outputs: List[Dict[str, List]]):
        """
        Aggregates the outputs of a Dataloader accross many batches.

        Parameters:

            outputs (List[Dict[str, List]]):
                A list of the outputs retrieved by processing each batch.
        """
        if not outputs:
            return {}

        keys = outputs[0].keys()
        aggregated = {k: sum([o[k].tolist() for o in outputs], []) for k in keys}

        return aggregated

    @staticmethod
    def aggregate_outputs(outputs, dataloader_names):
        """
        Aggregates eval outputs from multiple dataloaders (or a single one)
        into a dictionary in the following format:

        ```
        {
            data_loader_name: {
                output_item_1: [ ... ],  # list of all batches' outputs
                output_item_2: [ ... ],
            },
            ...
        }
        ```
        """
        if isinstance(outputs[0], list):
            assert len(dataloader_names) == len(outputs)

            return {
                k: BaseSeq2SeqModelV2.aggregate_single_output(outputs[i])
                for i, k in enumerate(dataloader_names)
            }

        return {dataloader_names[0]: BaseSeq2SeqModelV2.aggregate_single_output(outputs)}
