import os
import unittest

from typing import Dict
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch, call, ANY

from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from data_qa import XQUADDataModule


RUN_SLOW = os.getenv("RUN_SLOW", "0") == "1"


def create_mock_data(
    premise_column="premise", hypothesis_column="hypothesis", label_column="label"
) -> Dataset:
    return Dataset.from_dict(
        {
            premise_column: ["premise_1", "premise_2", "premise_3"] * 10,
            hypothesis_column: ["hypothesis_1", "hypothesis_2", "hypothesis_3"] * 10,
            label_column: [0, 1, 2] * 10,
        }
    )


class XQUADTestCase(TestCase):
    def test_init(self):
        module = XQUADDataModule(
            "google/mt5-small",
            384,
            128,
            32,
            "max_length",
            splits={"train": "custom", "validation": "another"},
            dataloader_num_workers=5,
        )

        self.assertIsInstance(module.tokenizer, PreTrainedTokenizerFast)

        self.assertEqual(module.max_length, 384)
        self.assertEqual(module.max_target_length, 128)
        self.assertEqual(module.batch_size, 32)
        self.assertEqual(module.collate_fn.padding, "max_length")
        self.assertEqual(module.splits["train"], "custom")
        self.assertEqual(module.splits["validation"], "another")
        self.assertEqual(module.dataloader_num_workers, 5)

    def test_prepare_data(self):
        module = XQUADDataModule("google/mt5-small", 10, 10, 32)
        module.download_data = Mock()

        module.prepare_data()

        module.download_data.assert_called_once()

    @patch("data_qa.load_dataset")
    def test_prepare_datasets(self, mock_load: Mock):
        train_dataset = Dataset.from_dict({"type": ["train"]})
        valid_datasets = [
            Dataset.from_dict({"type": ["valid"], "lang": [l]})
            for l in XQUADDataModule.XQUAD_LANGUAGES
        ]

        mock_load.side_effect = [train_dataset] + valid_datasets

        module = XQUADDataModule("google/mt5-small", 10, 10, 32)
        datasets = module.prepare_datasets()

        self.assertIn("train", datasets)
        self.assertIn("validation", datasets)

        self.assertIsInstance(datasets["train"], Dataset)
        self.assertIsInstance(datasets["validation"], DatasetDict)
        self.assertListEqual(
            list(datasets["validation"].keys()), XQUADDataModule.XQUAD_LANGUAGES
        )

    def test_prepare_input_sentence(self):
        dataset = Dataset.from_dict(
            {
                "answers": [{"answer_start": [0], "text": ["answer", "answer 2"]}],
                "context": ["This is supposed to be a very large context"],
                "id": ["56beb4343aeaaa14008c925c"],
                "question": ["Would you ask me?"],
            }
        )

        features = dataset.map(XQUADDataModule.prepare_input_sentence)
        inputs = features["input"]
        targets = features["target"]

        self.assertEqual(
            inputs,
            [
                "question: Would you ask me? context: This is supposed to be a very large context"
            ],
        )
        self.assertEqual(targets, ["answer"])

    # def test_setup(self, tokenizer_mock):
    #     module = TextClassificationDataModule("pretrained", 10, 10, 32)

    #     module.prepare_datasets = Mock(
    #         return_value={"train": ["train"], "validation": ["val"], "test": ["tst"]}
    #     )
    #     module.preprocess = Mock(side_effect=[["features_train"], ["features_val"], ["features_tst"]])
    #     module.prepare_features_for_model = Mock()

    #     module.setup("fit")

    #     module.prepare_datasets.assert_called_once()
    #     module.preprocess.assert_has_calls(
    #         [call(["train"], "train"), call(["val"], "validation"), call(["tst"], "test")]
    #     )
    #     module.prepare_features_for_model.assert_has_calls(
    #         [call("train"), call("validation"), call("test")]
    #     )

    #     self.assertEqual(module.data["train"], ["train"])
    #     self.assertEqual(module.data["validation"], ["val"])
    #     self.assertEqual(module.data["test"], ["tst"])

    #     self.assertEqual(module.features["train"], ["features_train"])
    #     self.assertEqual(module.features["validation"], ["features_val"])
    #     self.assertEqual(module.features["test"], ["features_tst"])

    # @patch("data_base.DataLoader")
    # def test_train_dataloader(self, dataloader_mock: Mock, tokenizer_mock):
    #     module = TextClassificationDataModule(
    #         "pretrained", 10, 10, 24, dataloader_num_workers=2
    #     )
    #     features = {"feat_one": [15] * 24, "feat_two": [12] * 24}
    #     module.features["train"] = features

    #     module.train_dataloader()

    #     dataloader_mock.assert_called_once_with(
    #         features,
    #         batch_size=24,
    #         shuffle=True,
    #         collate_fn=module.collate_fn,
    #         num_workers=module.dataloader_num_workers,
    #     )

    # @patch("data_base.DataLoader")
    # def test_val_dataloader(self, dataloader_mock: Mock, *args):
    #     module = TextClassificationDataModule(
    #         "pretrained", 10, 10, 24, dataloader_num_workers=2
    #     )
    #     features = {"feat_one": [15] * 24, "feat_two": [12] * 24}
    #     module.features["validation"] = features

    #     module.val_dataloader()

    #     dataloader_mock.assert_called_once_with(
    #         features,
    #         batch_size=24,
    #         shuffle=False,
    #         collate_fn=module.collate_fn,
    #         num_workers=module.dataloader_num_workers,
    #     )

    # @patch("data_base.DataLoader")
    # def test_test_dataloader(self, dataloader_mock: Mock, *args):
    #     module = TextClassificationDataModule(
    #         "pretrained", 10, 10, 24, dataloader_num_workers=2
    #     )
    #     features = {"feat_one": [15] * 24, "feat_two": [12] * 24}
    #     module.features["test"] = features

    #     module.test_dataloader()

    #     dataloader_mock.assert_called_once_with(
    #         features,
    #         batch_size=24,
    #         shuffle=False,
    #         collate_fn=module.collate_fn,
    #         num_workers=module.dataloader_num_workers,
    #     )

    # def test_preprocess(self, tokenizer_mock: Mock):
    #     module = TextClassificationDataModule("pretrained", 10, 10, 32)
    #     dataset = create_mock_data()

    #     result = module.preprocess(dataset, "train")

    #     tokenizer_mock.assert_called()

    #     self.assertIsInstance(result, Dataset)
    #     self.assertEqual(len(result), len(dataset))

    # def test_input_sequences(self, tokenizer_mock):
    #     example = {"premise": "my premise", "hypothesis": "my hypothesis", "label": 2}

    #     expected_input = "xnli: premise: my premise hypothesis: my hypothesis"
    #     expected_target = "2"

    #     example = TextClassificationDataModule.prepare_input_sentence(example)

    #     self.assertIn("input", example)
    #     self.assertIn("target", example)

    #     self.assertEqual(example["input"], expected_input)
    #     self.assertEqual(example["target"], expected_target)

    # def test_input_sequence_on_map(self, tokenizer_mock):
    #     dataset_original = Dataset.from_dict(
    #         {
    #             "premise": ["premise no 1", "premise no 2", "premise no 3"],
    #             "hypothesis": ["hypothesis no 1", "hypothesis no 2", "hypothesis no 3"],
    #             "label": [0, 1, 2],
    #             "some_random_column": ["val 1", "val 2", "val 3"],
    #         }
    #     )

    #     expected_inputs = [
    #         "xnli: premise: premise no 1 hypothesis: hypothesis no 1",
    #         "xnli: premise: premise no 2 hypothesis: hypothesis no 2",
    #         "xnli: premise: premise no 3 hypothesis: hypothesis no 3",
    #     ]

    #     expected_targets = ["0", "1", "2"]

    #     preprocessed = dataset_original.map(
    #         TextClassificationDataModule.prepare_input_sentence
    #     )

    #     self.assertEqual(preprocessed["input"], expected_inputs)
    #     self.assertEqual(preprocessed["target"], expected_targets)

    # def test_tokenization(self, tokenizer_mock):
    #     examples = {
    #         "premise": ["my premise", "my premise 2"],
    #         "hypothesis": ["my hypothesis", "my hypothesis 2"],
    #         "label": [2, 1],
    #         "input": ["input_1", "input_2"],
    #         "target": ["2", "1"],
    #     }

    #     tokenizer = FakeTokenizer()
    #     example = TextClassificationDataModule.tokenize_sequences(
    #         examples, tokenizer, 10, 5
    #     )

    #     self.assertIn("input_ids", example)
    #     self.assertIn("attention_mask", example)
    #     self.assertIn("target_ids", example)

    #     self.assertEqual(len(example["input_ids"]), 2)
    #     self.assertEqual(len(example["input_ids"][0]), 10)

    #     self.assertEqual(len(example["target_ids"]), 2)
    #     self.assertEqual(len(example["target_ids"][0]), 5)

    # def test_multiple_dataloaders(self, tokenizer_mock):
    #     class _ConcreteClass(TextClassificationDataModule):
    #         @property
    #         def val_dataloader_names(self):
    #             return ["pt", "en"]

    #         @property
    #         def test_dataloader_names(self):
    #             return ["pt", "en", "es"]

    #         def prepare_datasets(self):
    #             train_data = create_mock_data()

    #             val_data_1 = create_mock_data()
    #             val_data_2 = create_mock_data()

    #             tst_data_1 = create_mock_data()
    #             tst_data_2 = create_mock_data()
    #             tst_data_3 = create_mock_data()

    #             val_data = DatasetDict({"pt": val_data_1, "en": val_data_2})
    #             tst_data = DatasetDict({"pt": tst_data_1, "en": tst_data_2, "es": tst_data_3})

    #             return {"train": train_data, "validation": val_data, "test": tst_data}

    #     module = _ConcreteClass("pretrained", 10, 10, 10)
    #     module.setup()

    #     val_dataloaders = module.val_dataloader()
    #     tst_dataloaders = module.test_dataloader()

    #     self.assertIsInstance(val_dataloaders, list)
    #     self.assertEqual(len(val_dataloaders), 2)

    #     self.assertIsInstance(tst_dataloaders, list)
    #     self.assertEqual(len(tst_dataloaders), 3)
