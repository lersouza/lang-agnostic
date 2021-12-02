from dataclasses import dataclass

from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding


@dataclass
class DataDef:
    dataset_name: str
    premise_column: str
    hypothesis_column: str
    label_column: str
    num_labels: int


def data_def_name(dataset_name, subdataset_name):
    return f"{dataset_name}:{subdataset_name}"


DATA_DEFS = {
    "glue:mnli": DataDef("mnli", "premise", "hypothesis", "label", 3),
    "assin2:None": DataDef("assin2", "premise", "hypothesis", "entailment_judgment", 2),
}


def extract_seq2seq_features(
    dataset_definition, tokenizer, max_length, target_max_length, dataset
):
    hypothesis = dataset_definition.hypothesis_column
    premise = dataset_definition.premise_column
    label_column = dataset_definition.label_column

    def _preprocess_sample(example):
        sentence = f"premise: {example[premise]}. hypothesis: {example[hypothesis]}."
        original_label = example[label_column]

        encoded = tokenizer(
            sentence,
            max_length=max_length,
            truncation=True,
            return_overflowing_tokens=False,
        )

        target_encoded = tokenizer(
            str(original_label),
            max_length=target_max_length,
            truncation=True,
            return_overflowing_tokens=False,
        )

        encoded["target_ids"] = target_encoded["input_ids"]
        encoded["label"] = original_label

        return encoded

    features = dataset.map(
        _preprocess_sample, batched=False, remove_columns=[hypothesis, premise]
    )

    return features


class NLIDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer_name: str,
        train_dataset: str,
        train_subdataset: str,
        validation_set: str,
        batch_size: int = 32,
        max_length: int = 256,
        target_max_length: int = 5,
        xlang_dataset_name: str = None,
        xlang_subdataset_name: str = None,
        xlang_validation_set: str = None,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.train_dataset = train_dataset
        self.train_subdataset = train_subdataset
        self.validation_set = validation_set

        self.data_def = DATA_DEFS[data_def_name(train_dataset, train_subdataset)]

        self.xlang_dataset_name = xlang_dataset_name
        self.xlang_subdataset_name = xlang_subdataset_name
        self.xlang_validation_set = xlang_validation_set

        self.xlang_data_def = (
            DATA_DEFS[data_def_name(xlang_dataset_name, xlang_subdataset_name)]
            if xlang_dataset_name
            else None
        )

        self.batch_size = batch_size
        self.max_length = max_length
        self.target_max_length = target_max_length

        self.collate_fn = DataCollatorWithPadding(self.tokenizer, padding="max_length", max_length=self.max_length)

    @property
    def train_size(self):
        return self.__train_size or 0

    def prepare_data(self) -> None:
        load_dataset(self.train_dataset, self.train_subdataset)

        if self.xlang_dataset_name:
            load_dataset(self.xlang_dataset_name, self.xlang_subdataset_name)

    def setup(self, stage):
        dataset = load_dataset(self.train_dataset, self.train_subdataset)

        features = extract_seq2seq_features(
            self.data_def,
            self.tokenizer,
            self.max_length,
            self.target_max_length,
            dataset,
        )

        self.train_dataset = features["train"]
        self.valid_dataset = features[self.validation_set]
        self.__train_size = len(self.train_dataset) // self.batch_size

        if self.xlang_dataset_name:
            xlang_dataset = load_dataset(
                self.xlang_dataset_name, self.xlang_subdataset_name
            )

            xlang_features = extract_seq2seq_features(
                self.xlang_data_def,
                self.tokenizer,
                self.max_length,
                self.target_max_length,
                xlang_dataset,
            )

            self.cross_valid_dataset = xlang_features[self.xlang_validation_set]

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, True)

    def val_dataloader(self):
        val_loader1 = self._create_dataloader(self.valid_dataset)

        if not self.xlang_dataset_name:
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
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=4,
            collate_fn=self.collate_fn
        )
