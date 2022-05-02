""" Data Modules for Question-Answering Tasks """
import abc

from typing import Dict

from datasets import Dataset, DatasetDict
from data_base import BaseSeq2SeqDataModuleV2


class QuestionAnsweringDataModule(BaseSeq2SeqDataModuleV2, abc.ABC):
    """
    Base class for Question-Answering Datasets
    """

    def preprocess(self, dataset: Dataset, subset: str):
        features = dataset.map(
            self.prepare_qa_input_sentence,
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
    def prepare_qa_input_sentence(example: Dict):
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


class SquadDataModule(QuestionAnsweringDataModule):
    """
    A datamodule for using SQuAD dataset.
    """

    def prepare_datasets(self) -> DatasetDict:
        """
        Prepare the TydiQA GoldP data.
        This method uses the same set for both validation and test.
        """
        squad = self.load_dataset("squad", split=["train", "validation"])

        return DatasetDict(
            {
                "train": squad["train"],
                "validation": squad["validation"],
                "test": squad["validation"],
            }
        )


class SberquadDataModule(QuestionAnsweringDataModule):
    """
    A data module for working with SberQuAD dataset;
    """

    def prepare_datasets(self) -> DatasetDict:
        return self.load_dataset("sberquad")


class TydiQAGoldPModule(QuestionAnsweringDataModule):
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

    @property
    def val_dataloader_names(self):
        return list(self.TYDIQA_LANGUAGES.values())

    @property
    def test_dataloader_names(self):
        return list(self.TYDIQA_LANGUAGES.values())

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

    @staticmethod
    def extract_language(example: Dict):
        """
        Extract the language of each sample in a new `lang` column.
        """
        raw_lang = example["id"].split("-")[0]
        example["language"] = TydiQAGoldPModule.TYDIQA_LANGUAGES[raw_lang]

        return example
