import os

from abc import abstractmethod
from collections import defaultdict
from datasets import Dataset, DatasetDict, Split, load_dataset
from numpy import isin
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.data.data_collator import DataCollatorWithPadding

from typing import Any, Dict, Union


class TextClassificationDataModule(LightningDataModule):
    """
    Represents a base class for Text-Classification datasets.

    This class provides input data in the format: `premise: <premise>. hypothesis: <hypothesis>`.
    A `label` attribute is also available for training and validation.
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

        self.splits = {"train": "train", "validation": "validation"}
        self.splits.update(splits or {})

        self.data: Dict[str, Dataset] = {}
        self.features: Dict[str, Union[Dataset, DatasetDict]] = {}

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.collate_fn = DataCollatorWithPadding(
            self.tokenizer, padding=padding, max_length=max_length, return_tensors="pt"
        )

        self.save_hyperparameters()

    @property
    def premise_attr(self):
        return "premise"

    @property
    def hypothesis_attr(self):
        return "hypothesis"

    @property
    def label_attr(self):
        return "label"

    @property
    def model_features(self):
        return ["input_ids", "attention_mask", "target_ids", "label"]

    @property
    def val_dataloader_names(self):
        return ["default"]

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.features["train"], shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if isinstance(self.features["validation"], DatasetDict):
            return [
                self._create_dataloader(self.features["validation"][s])
                for s in self.val_dataloader_names
            ]

        return self._create_dataloader(self.features["validation"])

    @abstractmethod
    def prepare_datasets(self):
        raise NotImplementedError()

    def preprocess(self, dataset: Dataset, subset: str):
        features = dataset.map(
            self.prepare_input_sentence,
            desc="Formatting input sequence",
            fn_kwargs={
                "premise_attr": self.premise_attr,
                "hypothesis_attr": self.hypothesis_attr,
                "label_attr": self.label_attr,
            },
        )

        features = features.map(
            self.tokenize_sequences,
            batched=True,
            desc="Tokenizing",
            remove_columns=["input", "target"],
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "max_length": self.max_length,
                "max_target_length": self.max_target_length,
            },
        )

        return features

    def prepare_features_for_model(self, split: str):
        self.features[split].set_format(columns=self.model_features)

    def prepare_data(self) -> None:
        self.prepare_datasets()  # Force data download in prepare_data

    def setup(self, stage: str = None) -> None:
        self.data = self.prepare_datasets()

        for split in ("train", "validation"):
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

    @staticmethod
    def prepare_input_sentence(
        example,
        premise_attr: str = "premise",
        hypothesis_attr: str = "hypothesis",
        label_attr: str = "label",
    ):
        return {
            "input": f"xnli: premise: {example[premise_attr]} hypothesis: {example[hypothesis_attr]}",
            "target": str(example[label_attr]),
        }

    @staticmethod
    def tokenize_sequences(
        example: Any,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        max_target_length: int,
    ):

        model_inputs = tokenizer(
            example["input"], max_length=max_length, truncation=True
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                example["target"], max_length=max_target_length, truncation=True
            )
        model_inputs["target_ids"] = labels["input_ids"]

        return model_inputs


class Assin2DataModule(TextClassificationDataModule):
    def prepare_datasets(self):
        dataset = load_dataset(
            "assin2", split=[self.splits["train"], self.splits["validation"]]
        )

        train = dataset[0].rename_column("entailment_judgment", "label")
        valid = dataset[1].rename_column("entailment_judgment", "label")

        return {"train": train, "validation": valid}


class XnliDataModule(TextClassificationDataModule):
    XNLI_LANGUAGES = [
        "ar",
        "bg",
        "de",
        "el",
        "en",
        "es",
        "fr",
        "hi",
        "ru",
        "sw",
        "th",
        "tr",
        "ur",
        "vi",
        "zh",
    ]

    def __init__(self, *args, train_language: str = "en", **kwargs):
        super().__init__(*args, **kwargs)

        self.train_language = train_language

        self.save_hyperparameters()

    @property
    def val_dataloader_names(self):
        return self.XNLI_LANGUAGES

    def prepare_datasets(self):
        xnli_splits = [self.splits["train"], self.splits["validation"]]
        xnli_dataset = load_dataset("xnli", "all_languages", split=xnli_splits)

        train, valid = xnli_dataset
        filter_train_data = lambda e: e["language"] == self.train_language

        train = train.map(self.flatten, batched=True).filter(filter_train_data)
        valid = self._build_language_validation(valid.map(self.flatten, batched=True))

        return {"train": train, "validation": valid}

    def _build_language_validation(self, xnli_validation: Dataset):
        # We load a dataset for each language available in XNLI
        return DatasetDict(
            {
                lang: xnli_validation.filter(lambda e: e["language"] == lang)
                for lang in self.XNLI_LANGUAGES
            }
        )

    @staticmethod
    def flatten(examples):
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
