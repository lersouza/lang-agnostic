""" A base module for creating Seq2Seq data modules. """

import os

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

from datasets import Dataset, DatasetDict, load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
)


class BaseSeq2SeqDataModule(LightningDataModule, ABC):
    """
    Represents a base class for Seq-to-Seq datasets.
    """

    ALL_SPLITS = ["train", "validation", "test"]

    def __init__(
        self,
        tokenizer_name: str,
        max_length: int,
        max_target_length: int,
        batch_size: int,
        padding: str = "longest",
        splits: Dict[str, str] = None,
        dataloader_num_workers: int = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        train_language: str = None,
    ):
        """
        Params:
            tokenizer_name (`str`):
                The name of the tokenizer to be used.
            max_length (`int`):
                The maximum lenght for model input.
            max_target_length (`int`):
                The maximum length for the expected output.
            batch_size (`int`):
                The number of instances to be included in a single batch.
            padding (`str`):
                The Huggingface Tokenizer's padding strategy for a batch.
            splits (`Dict[str, str]`):
                A mapping for the actual splits to be used from `datasets.load_dataset(split=)`.
                For instance: if one would like to train in 20% of the dataset train set, the train
                split would be train[:20%], so:
                    `split={"train": "train[:20%]"}`.
            dataloader_num_workers (`int`, default: None):
                The number of workers to be used by the data loaders.
            cache_dir (`str`, default: `$HOME/.cache`):
                The cache directory to store the dataset.
            keep_in_memory (`bool`, default: `False`):
                Whether or not to keep the whole dataset in memory.
                Note: If keep_in_memory is set to `True`, Dataset cache will not be used.
            train_language (`str`, default: `None`):
                Specifies the language to be used for training. If a child module supports
                training in different languages, it will use this attribute
                to check which language to use.
        """
        super().__init__()

        self.max_length = max_length
        self.max_target_length = max_target_length
        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers or os.cpu_count()

        self.splits = {"train": "train", "validation": "validation", "test": "test"}
        self.splits.update(splits or {})

        self.data: Dict[str, Dataset] = {}
        self.features: Dict[str, Union[Dataset, DatasetDict]] = {}

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.collate_fn = DataCollatorWithPadding(
            self.tokenizer, padding=padding, max_length=max_length, return_tensors="pt"
        )

        self.cache_dir = cache_dir
        self.keep_in_memory = keep_in_memory

        self.train_language = train_language

        self.save_hyperparameters()

    @property
    def model_features(self):
        """
        Returns a list of attributes from the dataset that should be included in the batches.
        """
        return ["input_ids", "attention_mask", "target_ids"]

    @property
    def val_dataloader_names(self):
        """A list containing the names of the dataloaders returned by `val_dataloader` method."""
        return ["default"]

    @property
    def test_dataloader_names(self):
        """A list containing the names of the dataloaders returned by `test_dataloader` method."""
        return ["default"]

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.features["train"], shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._create_multiple_dataloaders(
            "validation", self.val_dataloader_names
        )

    def test_dataloader(self) -> DataLoader:
        return self._create_multiple_dataloaders("test", self.test_dataloader_names)

    def predict_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    @abstractmethod
    def prepare_datasets(self) -> DatasetDict:
        """
        Returns a `DatasetDict` with train, validation and test splits to be used.

        The data for each split should be raw data that will be processed
        by the `preprocess` method.
        """
        raise NotImplementedError()

    def load_dataset(
        self,
        name: str,
        subset: str = None,
        split: Union[str, List[str]] = None,
    ) -> DatasetDict:
        """
        Utility function to load a dataset from Huggingface's datasets library.

        Parameters:

            name (`str`):
                The name of the dataset to load

            subset (`str`, optional, Default = None):
                The subset name to load. For instance, `load_dataset("xnli", "all_languages")`

            split (`str` or `List[str]`. Defaults to `BaseSeq2SeqDataModule.ALL_SPLITS`):
                The name of the splits (`train`, `validation`, `test`) to load for the dataset.
                The names may be translated before passed to the HF's `load_dataset` function
                based on `self.splits` mapping passed through this class constructor.

                For instance, `validation` split may be translated into `validation_matched`
                if `self.splits["validation"]` is set to `"validation_matched"`.

        """
        split = split or BaseSeq2SeqDataModule.ALL_SPLITS

        datamodule_splits = split if isinstance(split, list) else [split]
        actual_splits = [self.splits[s] for s in datamodule_splits]

        datasets = load_dataset(
            name,
            subset,
            split=actual_splits,
            cache_dir=self.cache_dir,
            keep_in_memory=self.keep_in_memory,
        )

        return DatasetDict({s: datasets[i] for i, s in enumerate(split)})

    def download_data(self):
        """
        Utility method for derived classes in order to only download the datasets.
        This method is called during `prepare_data` hook.
        """

    def preprocess(self, dataset: Dataset, subset: str):
        """
        Subclasses of this may implement this method in order to provide
        proper handling of data to pass it to the model (tokenize, etc).
        """
        # pylint: disable=unused-argument
        # subset is an argument that may be useful for derived classes

        return dataset

    def prepare_features_for_model(self, split: str):
        """
        Format the `self.features` attribute by selecting the appropriate columns to be used
        while forming batches (based on `self.model_attributes` property).
        """
        self.features[split].set_format(columns=self.model_features)

    def prepare_data(self) -> None:
        self.download_data()

    def setup(self, stage: str = None) -> None:
        self.data = self.prepare_datasets()

        for split in ("train", "validation", "test"):
            self.features[split] = self.preprocess(self.data[split], split)
            self.prepare_features_for_model(split)

    def _create_dataloader(self, feature_set: Dataset, shuffle: bool = False):
        return DataLoader(
            feature_set,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.dataloader_num_workers,
        )

    def _create_multiple_dataloaders(
        self, split_name: str, dataloader_names: List[str], shuffle: bool = False
    ) -> Union[DataLoader, List[DataLoader]]:
        """
        Creates one or many dataloaders,
        depending on the content of features associated with `split_name`.
        """
        feature_set = self.features[split_name]

        if isinstance(feature_set, DatasetDict):
            return [
                self._create_dataloader(feature_set[s], shuffle)
                for s in dataloader_names
            ]

        return self._create_dataloader(feature_set, shuffle)

    @staticmethod
    def tokenize_sequences(
        example: Any,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        max_target_length: int,
        input_column_name: str = "input",
        target_column_name: str = "target",
    ):
        """
        Tokenizes a seq-2-seq input. The model input must be formatted in `input_column_name`.
        The model expected output must be formatted in `target_column_name`.
        """

        model_inputs = tokenizer(
            example[input_column_name], max_length=max_length, truncation=True
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                example[target_column_name],
                max_length=max_target_length,
                truncation=True,
            )
        model_inputs["target_ids"] = labels["input_ids"]

        return model_inputs


class BaseSeq2SeqDataModuleV2(LightningDataModule, ABC):
    """
    Represents a base class for Seq-to-Seq datasets.
    """

    ALL_SPLITS = ["train", "validation", "test"]

    def __init__(
        self,
        tokenizer_name: str,
        max_length: int,
        max_target_length: int,
        batch_size: int,
        padding: str = "longest",
        splits: Dict[str, str] = None,
        dataloader_num_workers: int = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        train_language: str = None,
        validate_on: List[str] = None,
    ):
        """
        Params:
            tokenizer_name (`str`):
                The name of the tokenizer to be used.
            max_length (`int`):
                The maximum lenght for model input.
            max_target_length (`int`):
                The maximum length for the expected output.
            batch_size (`int`):
                The number of instances to be included in a single batch.
            padding (`str`):
                The Huggingface Tokenizer's padding strategy for a batch.
            splits (`Dict[str, str]`):
                A mapping for the actual splits to be used from `datasets.load_dataset(split=)`.
                For instance: if one would like to train in 20% of the dataset train set, the train
                split would be train[:20%], so:
                    `split={"train": "train[:20%]"}`.
            dataloader_num_workers (`int`, default: None):
                The number of workers to be used by the data loaders.
            cache_dir (`str`, default: `$HOME/.cache`):
                The cache directory to store the dataset.
            keep_in_memory (`bool`, default: `False`):
                Whether or not to keep the whole dataset in memory.
                Note: If keep_in_memory is set to `True`, Dataset cache will not be used.
            train_language (`str`, default: `None`):
                Specifies the language to be used for training. If a child module supports
                training in different languages, it will use this attribute
                to check which language to use.
        """
        super().__init__()

        self.max_length = max_length
        self.max_target_length = max_target_length
        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers or os.cpu_count()

        self.splits = {"train": "train", "validation": "validation", "test": "test"}
        self.splits.update(splits or {})

        self.data: Dict[str, Dataset] = {}
        self.features: Dict[str, Union[Dataset, DatasetDict]] = {}

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.collate_fn = DataCollatorForSeq2Seq(
            self.tokenizer, padding=padding, max_length=max_length, return_tensors="pt"
        )

        self.cache_dir = cache_dir or Path.home() / ".cache"
        self.keep_in_memory = keep_in_memory

        self.train_language = train_language
        self.eval_subsets = {}

        if validate_on:
            self.eval_subsets["validation"] = validate_on

        self.save_hyperparameters()

    @property
    def model_features(self):
        """
        Returns a list of attributes from the dataset that should be included in the batches.
        """
        return ["input_ids", "attention_mask", "labels"]

    @property
    def dataloader_names(self) -> Dict[str, List[str]]:
        """
        Return the names of the dataloaders used for validation and test.
        The result is a Dictionary with the following keys:

            validation: a list of names for validation data loaders
            test:  a list of names for validation data loaders

        The difference from `dataloader_names` to `supported_dataloader_names` is that the former
        returns a list of the dataloaders that will actually be used by `val_dataloader` or
        `test_dataloader` methods, while the latter return all supported dataloader names by
        the module.
        """
        dataloader_names = self.supported_dataloader_names
        dataloader_names.update(self.eval_subsets)

        return dataloader_names

    @property
    def supported_dataloader_names(self) -> Dict[str, List[str]]:
        """
        Return the names of all available dataloaders for validation and test.
        The result is a Dictionary with the following keys:

            validation: a list of names for validation data loaders
            test:  a list of names for validation data loaders
        """
        return {"validation": ["default"], "test": ["default"]}

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.features["train"], shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._create_multiple_dataloaders("validation")

    def test_dataloader(self) -> DataLoader:
        return self._create_multiple_dataloaders("test")

    def predict_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    @abstractmethod
    def prepare_datasets(self) -> DatasetDict:
        """
        Returns a `DatasetDict` with train, validation and test splits to be used.

        The data for each split should be raw data that will be processed
        by the `preprocess` method.
        """
        raise NotImplementedError()

    def load_dataset(
        self,
        name: str,
        subset: str = None,
        split: Union[str, List[str]] = None,
        **kwargs,
    ) -> DatasetDict:
        """
        Utility function to load a dataset from Huggingface's datasets library.

        Parameters:

            name (`str`):
                The name of the dataset to load

            subset (`str`, optional, Default = None):
                The subset name to load. For instance, `load_dataset("xnli", "all_languages")`

            split (`str` or `List[str]`. Defaults to `BaseSeq2SeqDataModule.ALL_SPLITS`):
                The name of the splits (`train`, `validation`, `test`) to load for the dataset.
                The names may be translated before passed to the HF's `load_dataset` function
                based on `self.splits` mapping passed through this class constructor.

                For instance, `validation` split may be translated into `validation_matched`
                if `self.splits["validation"]` is set to `"validation_matched"`.

            kwargs (Dict[str, Any]):
                Any other additional args to be passed to `datasets.load_dataset` method.

        """
        split = split or BaseSeq2SeqDataModule.ALL_SPLITS

        datamodule_splits = split if isinstance(split, list) else [split]
        actual_splits = [self.splits[s] for s in datamodule_splits]

        datasets = load_dataset(
            name,
            subset,
            split=actual_splits,
            cache_dir=self.cache_dir,
            keep_in_memory=self.keep_in_memory,
            **kwargs,
        )

        return DatasetDict({s: datasets[i] for i, s in enumerate(split)})

    def download_data(self):
        """
        Utility method for derived classes in order to only download the datasets.
        This method is called during `prepare_data` hook.
        """

    def preprocess(self, dataset: Dataset, subset: str):
        """
        Subclasses of this may implement this method in order to provide
        proper handling of data to pass it to the model (tokenize, etc).
        """
        # pylint: disable=unused-argument
        # subset is an argument that may be useful for derived classes

        return dataset

    def prepare_features_for_model(self, split: str):
        """
        Format the `self.features` attribute by selecting the appropriate columns to be used
        while forming batches (based on `self.model_attributes` property).
        """
        self.features[split].set_format(columns=self.model_features)

    def prepare_data(self) -> None:
        self.download_data()

    def setup(self, stage: str = None) -> None:
        self.data = self.prepare_datasets()

        for split in ("train", "validation", "test"):
            self.features[split] = self.preprocess(self.data[split], split)
            self.prepare_features_for_model(split)

    def _create_dataloader(self, feature_set: Dataset, shuffle: bool = False):
        return DataLoader(
            feature_set,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.dataloader_num_workers,
        )

    def _create_multiple_dataloaders(
        self, split_name: str, shuffle: bool = False
    ) -> Union[DataLoader, List[DataLoader]]:
        """
        Creates one or many dataloaders,
        depending on the content of features associated with `split_name`.
        """
        feature_set = self.features[split_name]
        eval_subset = self.dataloader_names[split_name]

        if isinstance(feature_set, DatasetDict):
            return [
                self._create_dataloader(feature_set[s], shuffle) for s in eval_subset
            ]

        return self._create_dataloader(feature_set, shuffle)

    @staticmethod
    def tokenize_sequences(
        example: Any,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        max_target_length: int,
        input_column_name: str = "input",
        target_column_name: str = "target",
    ):
        """
        Tokenizes a seq-2-seq input. The model input must be formatted in `input_column_name`.
        The model expected output must be formatted in `target_column_name`.
        """

        model_inputs = tokenizer(
            example[input_column_name], max_length=max_length, truncation=True
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                example[target_column_name],
                max_length=max_target_length,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    @staticmethod
    def filter_data_by_lang(example: Dict, language: str):
        """
        Helper function to filter a Dataset by language.
        """
        return example["language"] == language
