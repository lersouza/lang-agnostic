import os

from typing import Any, Dict, List, Union

from datasets import Dataset, DatasetDict
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizer


class BaseSeq2SeqDataModule(LightningDataModule):
    """
    Represents a base class for Seq-to-Seq datasets.
    """

    def __init__(
        self,
        tokenizer_name: str,
        max_length: int,
        max_target_length: int,
        batch_size: int,
        padding: str = "longest",
        splits: Dict[str, str] = None,
        dataloader_num_workers: int = None,
    ):
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

        self.save_hyperparameters()

    @property
    def all_splits(self):
        return [self.splits[i] for i in ("train", "validation", "test")]

    @property
    def model_features(self):
        return ["input_ids", "attention_mask", "target_ids"]

    @property
    def val_dataloader_names(self):
        return ["default"]

    @property
    def test_dataloader_names(self):
        return ["default"]

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.features["train"], shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._create_multiple_dataloaders(
            "validation", self.val_dataloader_names
        )

    def test_dataloader(self) -> DataLoader:
        return self._create_multiple_dataloaders("test", self.test_dataloader_names)

    def prepare_datasets(self):
        raise NotImplementedError()

    def download_data(self):
        """
        Utility method for derived classes in order to only download the datasets.
        This method is called during `prepare_data` hook.
        """
        pass

    def preprocess(self, dataset: Dataset, subset: str):
        """
        Subclasses of this may implement this method in order to provide
        proper handling of data to pass it to the model (tokenize, etc).
        """
        return dataset

    def prepare_features_for_model(self, split: str):
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
