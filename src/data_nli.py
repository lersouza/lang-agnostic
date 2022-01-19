import os

from abc import abstractmethod
from collections import defaultdict
from datasets import Dataset, load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.data.data_collator import DataCollatorWithPadding

from typing import Any, Dict


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

        self.max_length = max_length
        self.max_target_length = max_target_length
        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers or os.cpu_count()

        self.splits = {"train": "train", "validation": "validation"}
        self.splits.update(splits or {})

        self.data: Dict[str, Dataset] = {}
        self.features: Dict[str, Dataset] = {}

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.collate_fn = DataCollatorWithPadding(
            self.tokenizer, padding=padding, max_length=max_length, return_tensors="pt"
        )

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

    def split(self, split_name):
        return self.splits.get(split_name, split_name)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.features["train"],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.dataloader_num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.features["validation"],
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.dataloader_num_workers,
        )

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
            remove_columns=["input", "label_string"],
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "max_length": self.max_length,
                "max_target_length": self.max_target_length,
            },
        )

        return features

    def prepare_features_for_model(self, subset: str):
        self.features[subset].set_format(columns=self.model_features)

    def prepare_data(self) -> None:
        self.prepare_datasets()  # Force data download in prepare_data

    def setup(self, stage: str = None) -> None:
        self.data = self.prepare_datasets()

        for subset in ("train", "validation"):
            self.features[subset] = self.preprocess(self.data[subset], subset)
            self.prepare_features_for_model(subset)

    @staticmethod
    def prepare_input_sentence(
        example,
        premise_attr: str = "premise",
        hypothesis_attr: str = "hypothesis",
        label_attr: str = "label",
    ):
        return {
            "input": f"premise: {example[premise_attr]}. hypothesis: {example[hypothesis_attr]}",
            "label_string": str(example[label_attr]),
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
                example["label_string"], max_length=max_target_length, truncation=True
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
    def __init__(self, *args, languages: Dict[str, str] = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.languages = {"train": "en", "validation": "all_languages"}
        self.languages.update(languages or {})

    def prepare_datasets(self):
        train = load_dataset(
            "xnli", self.languages["train"], split=self.splits["train"]
        )
        valid = load_dataset(
            "xnli", self.languages["validation"], split=self.splits["validation"]
        )

        if self.languages["train"] == "all_languages":
            train = train.map(
                self.flatten, batched=True, desc="Flatenning multi language dataset"
            )

        if self.languages["validation"] == "all_languages":
            valid = valid.map(
                self.flatten, batched=True, desc="Flatenning multi language dataset"
            )

        return {"train": train, "validation": valid}

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


if __name__ == "__main__":
    from random import randint

    data_module = Assin2DataModule(
        "google/byt5-small",
        1024,
        5,
        32,
        train_split="train",
        validation_split="validation",
    )
    data_module.prepare_data()
    data_module.setup("fit")

    batch = next(iter(data_module.val_dataloader()))

    print("Attrs:", batch.keys())
    print("Shape:", batch["input_ids"].shape)
    print("Label sample:")

    idx = randint(0, 32)

    print("\tOriginal:", data_module.data["validation"][idx]["label"])
    print("\tBatch:", batch["labels"][idx])
