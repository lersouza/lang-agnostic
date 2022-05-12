""" Data modules for NLI Task """

import abc

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, DatasetDict

from data_base import BaseSeq2SeqDataModuleV2


class TextClassificationDataModule(BaseSeq2SeqDataModuleV2, abc.ABC):
    """
    Represents a base class for Text-Classification datasets.

    This class provides input data in the format: `premise: <premise>. hypothesis: <hypothesis>`.
    A `label` attribute is also available for training and validation.
    """

    @property
    def premise_attr(self):
        """The name of the premise column in the dataset."""
        return "premise"

    @property
    def hypothesis_attr(self):
        """The name of the hypothesis column in the dataset."""
        return "hypothesis"

    @property
    def label_attr(self):
        """The name of the label column in the dataset."""
        return "label"

    @property
    def model_features(self):
        return ["input_ids", "attention_mask", "target_ids", "label"]

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

    @staticmethod
    def prepare_input_sentence(
        example,
        premise_attr: str = "premise",
        hypothesis_attr: str = "hypothesis",
        label_attr: str = "label",
    ):
        """
        Creates an `input` and `target` columns in the example for holding formatted
        model input and expected output.

        Input is formatted as `xnli: premise: <<premise>> hypothesis: <<hypothesis>>.
        Target is a string representing the label id: Ex: `"0"`.
        """
        return {
            "input": (
                f"xnli: premise: {example[premise_attr]} "
                f"hypothesis: {example[hypothesis_attr]}"
            ),
            "target": str(example[label_attr]),
        }


class Assin2DataModule(TextClassificationDataModule):
    """A data module for the ASSIN2 Portuguese Dataset."""

    def prepare_datasets(self):
        datap = str(Path(__file__).parents[1].absolute() / "datasets/assin2.py")
        dataset = self.load_dataset(datap).rename_column("entailment_judgment", "label")

        return dataset


class XnliDataModule(TextClassificationDataModule):
    """A data module for the XNLI dataset."""

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

    @property
    def supported_dataloader_names(self) -> Dict[str, List[str]]:
        return self.XNLI_LANGUAGES

    def prepare_datasets(self):
        xnli_dataset = self.load_dataset("xnli", "all_languages")
        xnli_dataset = xnli_dataset.map(self.flatten, batched=True)

        lang_to_train = self.train_language or "en"

        xnli_dataset["train"] = xnli_dataset["train"].filter(
            self.filter_data_by_lang, fn_kwargs={"language": lang_to_train}
        )
        xnli_dataset["validation"] = self._build_lang_valid_set(
            xnli_dataset["validation"]
        )
        xnli_dataset["test"] = self._build_lang_valid_set(xnli_dataset["test"])

        return xnli_dataset

    def _build_lang_valid_set(self, xnli_validation: Dataset):
        # We load a dataset for each language available in XNLI
        return DatasetDict(
            {
                lang: xnli_validation.filter(
                    self.filter_data_by_lang, fn_kwargs={"language": lang}
                )
                for lang in self.XNLI_LANGUAGES
            }
        )

    @staticmethod
    def filter_data_by_lang(example: Dict, language: str):
        """
        Helper function to filter a Dataset by language.
        """
        return example["language"] == language

    @staticmethod
    def flatten(examples):
        """
        XNLI, according to https://huggingface.co/datasets/xnli#data-instances,
        has the following format:
        {
            "hypothesis": {
                \"language\": [\"ar\", \"bg\", \"de\", \"el\", \"en\", \"es\", \"fr\", \"hi\",
                               \"ru\", \"sw\", \"th\", \"tr\", \"ur\", \"vi\", \"zh\"],
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
            'premise': 'you know during the season and i guess at at ...
        }
        """
        hypothesis = [
            (i, lang, trans)
            for i, ex in enumerate(examples["hypothesis"])
            for lang, trans in zip(ex["language"], ex["translation"])
        ]
        premises = [
            (i, lang, prem)
            for i, ex in enumerate(examples["premise"])
            for lang, prem in ex.items()
        ]

        features = defaultdict(list)

        for (i, lang, hypo), (_, _, prem) in zip(hypothesis, premises):

            features["premise"].append(prem)
            features["hypothesis"].append(hypo)
            features["language"].append(lang)
            features["label"].append(examples["label"][i])

        return features
