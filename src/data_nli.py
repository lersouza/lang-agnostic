import abc

from collections import defaultdict
from datasets import Dataset, DatasetDict

from data_base import BaseSeq2SeqDataModule


class TextClassificationDataModule(BaseSeq2SeqDataModule, abc.ABC):
    """
    Represents a base class for Text-Classification datasets.

    This class provides input data in the format: `premise: <premise>. hypothesis: <hypothesis>`.
    A `label` attribute is also available for training and validation.
    """

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
        return {
            "input": (
                f"xnli: premise: {example[premise_attr]} "
                f"hypothesis: {example[hypothesis_attr]}"
            ),
            "target": str(example[label_attr]),
        }


class Assin2DataModule(TextClassificationDataModule):
    def prepare_datasets(self):
        dataset = self.load_dataset("assin2")
        dataset.rename_column_("entailment_judgment", "label")

        return dataset


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

    @property
    def test_dataloader_names(self):
        return self.XNLI_LANGUAGES

    def prepare_datasets(self):
        xnli_dataset = self.load_dataset("xnli", "all_languages")
        xnli_dataset = xnli_dataset.map(self.flatten, batched=True)

        filter_train_data = lambda e: e["language"] == self.train_language

        xnli_dataset["train"] = xnli_dataset["train"].filter(filter_train_data)
        xnli_dataset["validation"] = self._build_language_validation(xnli_dataset["validation"])
        xnli_dataset["test"] = self._build_language_validation(xnli_dataset["test"])

        return xnli_dataset

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
