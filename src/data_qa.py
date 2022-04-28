""" Data Modules for Question-Answering Tasks """
import logging

from typing import Any, Dict, List, Union

import torch

from datasets import Dataset, DatasetDict
from transformers import BatchEncoding

from data_base import BaseSeq2SeqDataModule


class TydiQAGoldPModule(BaseSeq2SeqDataModule):
    """
    Represents a data module for the GoldP task of the TydiQA dataset.
    """

    TYDIQA_LANGUAGES = {
        "finnish": "fi",
        "telugu": "te",
        "russian": "ru",
        "arabic": "ar",
        "indonesian": "id",
        "english": "en",
        "swahili": "sw",
        "korean": "ko",
        "bengali": "bn",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # We setup return tensors as None to enable the input to have the reference answers
        self.original_collator = self.collate_fn
        self.collate_fn = self.collate_wrapper

    @property
    def val_dataloader_names(self):
        return list(self.TYDIQA_LANGUAGES.values())

    @property
    def test_dataloader_names(self):
        return list(self.TYDIQA_LANGUAGES.values())

    @property
    def model_features(self):
        return ["input_ids", "attention_mask", "target_ids", "id", "answers"]

    def collate_wrapper(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        A wrapper functions that uses the default collator from the Base Data Module,
        but allow strings to be in the batch.

        After the default collator is used (which will not return tensors, but lists),
        the model input attributes (input_ids, attention_mask and target_ids) will be converted
        to tensors. The rest of the attributes will remain as is.
        """
        # pylint: disable=not-callable

        input_feat, target_feat = [], []
        additional = {"id": [], "answers": []}

        for feature in features:
            input_feat.append(
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                }
            )
            target_feat.append({"input_ids": feature["target_ids"]})

            additional["id"].append(feature["id"])
            additional["answers"].append(feature["answers"])

        input_collated = self.original_collator(input_feat)
        input_collated["target_ids"] = self.original_collator(target_feat)["input_ids"]
        input_collated.update(additional)

        return input_collated

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        # pylint: disable=no-member

        for k in ("input_ids", "attention_mask", "target_ids"):
            batch[k] = batch[k].to(device)

    def prepare_datasets(self) -> DatasetDict:
        """
        Prepare the TydiQA GoldP data.
        This method uses the same set for both validation and test.
        """
        tydi = self.load_dataset(
            "tydiqa", "secondary_task", split=["train", "validation"]
        )
        tydi = tydi.map(self.extract_language, batched=False)

        if self.train_language:
            tydi["train"] = tydi["train"].filter(
                self.filter_data_by_lang, fn_kwargs={"language": self.train_language}
            )

        valid = DatasetDict(
            {
                lang: tydi["validation"].filter(
                    self.filter_data_by_lang, fn_kwargs={"language": lang}
                )
                for lang in self.TYDIQA_LANGUAGES.values()
            }
        )

        return DatasetDict({"train": tydi["train"], "validation": valid, "test": valid})

    def preprocess(self, dataset: Dataset, subset: str):
        features = dataset.map(
            self.prepare_input_sentence,
            desc="Formatting input sequence",
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
    def extract_language(example: Dict):
        """
        Extract the language of each sample in a new `lang` column.
        """
        raw_lang = example["id"].split("-")[0]
        example["lang"] = TydiQAGoldPModule.TYDIQA_LANGUAGES[raw_lang]

        return example

    @staticmethod
    def filter_data_by_lang(example: Dict, language: str):
        """
        Helper function to filter a Dataset by language.
        """
        return example["lang"] == language

    @staticmethod
    def prepare_input_sentence(example: Dict):
        """
        Formats the input for the model as: `question: <<question>> context: <<context>>
        The target (for training) is the first available answer
        """
        ans = example["answers"]["text"]
        qas = example["question"]
        ctx = example["context"]

        example["input"] = f"question: {qas} context: {ctx}"
        example["target"] = ans[0]

        return example
