from typing import Dict
from datasets import Dataset, DatasetDict, load_dataset
from data_base import BaseSeq2SeqDataModule


class XQUADDataModule(BaseSeq2SeqDataModule):
    XQUAD_LANGUAGES = [
        "ar",
        "de",
        "el",
        "es",
        "en",
        "hi",
        "ro",
        "ru",
        "th",
        "tr",
        "vi",
        "zh",
    ]

    def prepare_datasets(self) -> None:
        train = load_dataset("squad", split=self.splits["train"])
        valid = DatasetDict(
            {
                lang: load_dataset(
                    "xquad", f"xquad.{lang}", split=self.splits["validation"]
                )
                for lang in self.XQUAD_LANGUAGES
            }
        )

        return {"train": train, "validation": valid}

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
    def prepare_input_sentence(example: Dict):
        ans = example["answers"]["text"]
        qas = example["question"]
        ctx = example["context"]

        example["input"] = f"question: {qas} context: {ctx}"
        example["target"] = ans[0]

        return example
