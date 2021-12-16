from collections import defaultdict
from dataclasses import dataclass
from types import FunctionType

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
    process_function: FunctionType


def data_def_name(dataset_name, subdataset_name):
    return f"{dataset_name}:{subdataset_name}"


"""
dataset = load_dataset("xnli", "mnli")
features = dataset.map(process_xnli, batched=False)

valid = load_dataset("xnli", "all_languages", split="validation")
"""


def format_input_sequence(example, premise_column, hypothesis_column):
    return (
        f"premise: {example[premise_column]}. hypothesis: {example[hypothesis_column]}."
    )


def flat_xnli(examples):
    """
    XNLI, according to https://huggingface.co/datasets/xnli#data-instances, has the following format:

    {
        "hypothesis": {
            \"language\": [\"ar\", \"bg\", \"de\", \"el\", \"en\", \"es\", \"fr\", \"hi\", \"ru\", \"sw\", \"th\", \"tr\", \"ur\", \"vi\", \"zh\"],
            \"translation\": [\"احد اع...",
        }
        "label": 0,
        "premise": {
            \"ar\": \"واحدة من رقابنا ستقوم بتنفيذ تعليماتك كلها بكل دقة\",
            \"bg\": \"един от нашите номера ще ви даде инструкции .\",
            \"de\": \"Eine ..."
        }
    }

    This function flattens the dataset, so every language gets its own examples:

    {
        'hypothesis': 'You lose the things to the following level if the people recall .',
        'label': 0,
        'language': 'en',
        'premise': 'you know during the season and i guess at at your level uh you lose them to the next ...
    }
    """
    h = [
        (i, lang, trans)
        for i, ex in enumerate(examples["hypothesis"])
        for lang, trans in zip(ex["language"], ex["translation"])
    ]
    p = [
        (i, lang, prem)
        for i, ex in enumerate(examples["premise"])
        for lang, prem in ex.items()
    ]

    features = defaultdict(list)

    for (i, l, hy), (_, _, pr) in zip(h, p):

        features["premise"].append(pr)
        features["hypothesis"].append(hy)
        features["language"].append(l)
        features["label"].append(examples["label"][i])

    return features


def process_xnli(_, tokenizer, max_length, target_max_length, dataset):
    """
    Process XNLI for seq2seq models. This functions extracts features in the form:

    {
        "input_ids": [ ... ]
        "attention_mask": [ ... ]
        "target_ids: [ ... ],
        "label": [ ... ]
    }

    The input ids are encoded tokens from the formatted string:

        premise: <premise>. hypothesis: <hypothesis>.
    """

    def _process_sample(examples):
        input_seq = [
            f"premise: {prem}. hypothesis: {hyp}."
            for prem, hyp in zip(examples["premise"], examples["hypothesis"])
        ]
        target_seq = [str(l) for l in examples["label"]]

        encoded = tokenizer(
            input_seq,
            max_length=max_length,
            truncation=True,
            return_overflowing_tokens=False,
        )

        target_encoded = tokenizer(
            target_seq,
            max_length=target_max_length,
            truncation=True,
            return_overflowing_tokens=False,
        )

        encoded["target_ids"] = target_encoded["input_ids"]
        encoded["label"] = examples["label"]

        return encoded

    examples = dataset.map(
        flat_xnli, batched=True, remove_columns=["hypothesis", "premise", "label"]
    )
    features = examples.map(
        _process_sample,
        batched=True,
        remove_columns=["premise", "hypothesis", "language"],
    )

    return features


def process_mnli(
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
        _preprocess_sample,
        batched=False,
        remove_columns=[hypothesis, premise, label_column],
    )

    return features


DATA_DEFS = {
    "glue:mnli": DataDef("mnli", "premise", "hypothesis", "label", 3, process_mnli),
    "assin2:None": DataDef("assin2", "premise", "hypothesis", "entailment_judgment", 2, process_xnli),
    "xnli:all_languages": DataDef("xnli", "premise", "hypothesis", "label", 3, process_xnli),
}


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
        **kwargs,
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

        self.collate_fn = DataCollatorWithPadding(
            self.tokenizer,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt",
        )

    @property
    def train_size(self):
        return self.__train_size or 0

    def prepare_data(self) -> None:
        load_dataset(self.train_dataset, self.train_subdataset)

        if self.xlang_dataset_name:
            load_dataset(self.xlang_dataset_name, self.xlang_subdataset_name)

    def setup(self, stage):
        dataset = load_dataset(self.train_dataset, self.train_subdataset)

        features = self.data_def.process_function(
            self.data_def,
            self.tokenizer,
            self.max_length,
            self.target_max_length,
            dataset,
        )

        self.__train_dataset_obj = features["train"]
        self.__valid_dataset_obj = features[self.validation_set]
        self.__train_size = len(self.__train_dataset_obj) // self.batch_size

        if self.xlang_dataset_name:
            xlang_dataset = load_dataset(
                self.xlang_dataset_name, self.xlang_subdataset_name
            )

            xlang_features = self.xlang_data_def.process_function(
                self.xlang_data_def,
                self.tokenizer,
                self.max_length,
                self.target_max_length,
                xlang_dataset,
            )

            self.__cross_valid_dataset_obj = xlang_features[self.xlang_validation_set]

    def train_dataloader(self):
        return self._create_dataloader(self.__train_dataset_obj, True)

    def val_dataloader(self):
        val_loader1 = self._create_dataloader(self.__valid_dataset_obj)

        if not self.xlang_dataset_name:
            return val_loader1

        val_loader2 = self._create_dataloader(self.__cross_valid_dataset_obj)

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
            collate_fn=self.collate_fn,
        )
