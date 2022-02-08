import os
import unittest

from typing import Dict
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch, call, ANY

from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader

from data_nli import Assin2DataModule, TextClassificationDataModule, XnliDataModule
from test_utils import FakeTokenizer


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


def create_xnli_mock_data() -> Dataset:
    return Dataset.from_dict(
        {
            "premise": [{"l1": "premise_l1", "l2": "premise_l2", "l3": "premise_l3"}]
            * 10,
            "hypothesis": [
                {"language": ["l1", "l2", "l3"], "translation": ["t1", "t2", "t3"]}
            ]
            * 10,
            "label": [0] * 10,
        }
    )


class MockedTextDataModule(TextClassificationDataModule):
    def prepare_datasets(self):
        pass  # Just to be used with mocks



@patch("data_base.AutoTokenizer.from_pretrained", return_value=FakeTokenizer())
class TextClassificationTest(TestCase):
    def test_init(self, tokenizer_mock: Mock):
        module = MockedTextDataModule(
            "pretrained-model",
            10,
            2,
            32,
            "max_length",
            splits={"train": "custom", "validation": "another"},
            dataloader_num_workers=5,
        )

        tokenizer_mock.assert_called_once_with("pretrained-model")

        self.assertEqual(module.max_length, 10)
        self.assertEqual(module.max_target_length, 2)
        self.assertEqual(module.batch_size, 32)
        self.assertEqual(module.collate_fn.padding, "max_length")
        self.assertEqual(module.splits["train"], "custom")
        self.assertEqual(module.splits["validation"], "another")
        self.assertEqual(module.dataloader_num_workers, 5)

    def test_prepare_data(self, tokenizer_mock):
        module = MockedTextDataModule("pretrained", 10, 10, 32)
        module.download_data = Mock()

        module.prepare_data()

        module.download_data.assert_called_once()

    def test_setup(self, tokenizer_mock):
        module = MockedTextDataModule("pretrained", 10, 10, 32)

        module.prepare_datasets = Mock(
            return_value={"train": ["train"], "validation": ["val"], "test": ["tst"]}
        )
        module.preprocess = Mock(
            side_effect=[["features_train"], ["features_val"], ["features_tst"]]
        )
        module.prepare_features_for_model = Mock()

        module.setup("fit")

        module.prepare_datasets.assert_called_once()
        module.preprocess.assert_has_calls(
            [
                call(["train"], "train"),
                call(["val"], "validation"),
                call(["tst"], "test"),
            ]
        )
        module.prepare_features_for_model.assert_has_calls(
            [call("train"), call("validation"), call("test")]
        )

        self.assertEqual(module.data["train"], ["train"])
        self.assertEqual(module.data["validation"], ["val"])
        self.assertEqual(module.data["test"], ["tst"])

        self.assertEqual(module.features["train"], ["features_train"])
        self.assertEqual(module.features["validation"], ["features_val"])
        self.assertEqual(module.features["test"], ["features_tst"])

    @patch("data_base.DataLoader")
    def test_train_dataloader(self, dataloader_mock: Mock, tokenizer_mock):
        module = MockedTextDataModule(
            "pretrained", 10, 10, 24, dataloader_num_workers=2
        )
        features = {"feat_one": [15] * 24, "feat_two": [12] * 24}
        module.features["train"] = features

        module.train_dataloader()

        dataloader_mock.assert_called_once_with(
            features,
            batch_size=24,
            shuffle=True,
            collate_fn=module.collate_fn,
            num_workers=module.dataloader_num_workers,
        )

    @patch("data_base.DataLoader")
    def test_val_dataloader(self, dataloader_mock: Mock, *args):
        module = MockedTextDataModule(
            "pretrained", 10, 10, 24, dataloader_num_workers=2
        )
        features = {"feat_one": [15] * 24, "feat_two": [12] * 24}
        module.features["validation"] = features

        module.val_dataloader()

        dataloader_mock.assert_called_once_with(
            features,
            batch_size=24,
            shuffle=False,
            collate_fn=module.collate_fn,
            num_workers=module.dataloader_num_workers,
        )

    @patch("data_base.DataLoader")
    def test_test_dataloader(self, dataloader_mock: Mock, *args):
        module = MockedTextDataModule(
            "pretrained", 10, 10, 24, dataloader_num_workers=2
        )
        features = {"feat_one": [15] * 24, "feat_two": [12] * 24}
        module.features["test"] = features

        module.test_dataloader()

        dataloader_mock.assert_called_once_with(
            features,
            batch_size=24,
            shuffle=False,
            collate_fn=module.collate_fn,
            num_workers=module.dataloader_num_workers,
        )

    def test_preprocess(self, tokenizer_mock: Mock):
        module = MockedTextDataModule("pretrained", 10, 10, 32)
        dataset = create_mock_data()

        result = module.preprocess(dataset, "train")

        tokenizer_mock.assert_called()

        self.assertIsInstance(result, Dataset)
        self.assertEqual(len(result), len(dataset))

    def test_input_sequences(self, tokenizer_mock):
        example = {"premise": "my premise", "hypothesis": "my hypothesis", "label": 2}

        expected_input = "xnli: premise: my premise hypothesis: my hypothesis"
        expected_target = "2"

        example = TextClassificationDataModule.prepare_input_sentence(example)

        self.assertIn("input", example)
        self.assertIn("target", example)

        self.assertEqual(example["input"], expected_input)
        self.assertEqual(example["target"], expected_target)

    def test_input_sequence_on_map(self, tokenizer_mock):
        dataset_original = Dataset.from_dict(
            {
                "premise": ["premise no 1", "premise no 2", "premise no 3"],
                "hypothesis": ["hypothesis no 1", "hypothesis no 2", "hypothesis no 3"],
                "label": [0, 1, 2],
                "some_random_column": ["val 1", "val 2", "val 3"],
            }
        )

        expected_inputs = [
            "xnli: premise: premise no 1 hypothesis: hypothesis no 1",
            "xnli: premise: premise no 2 hypothesis: hypothesis no 2",
            "xnli: premise: premise no 3 hypothesis: hypothesis no 3",
        ]

        expected_targets = ["0", "1", "2"]

        preprocessed = dataset_original.map(
            TextClassificationDataModule.prepare_input_sentence
        )

        self.assertEqual(preprocessed["input"], expected_inputs)
        self.assertEqual(preprocessed["target"], expected_targets)

    def test_tokenization(self, tokenizer_mock):
        examples = {
            "premise": ["my premise", "my premise 2"],
            "hypothesis": ["my hypothesis", "my hypothesis 2"],
            "label": [2, 1],
            "input": ["input_1", "input_2"],
            "target": ["2", "1"],
        }

        tokenizer = FakeTokenizer()
        example = TextClassificationDataModule.tokenize_sequences(
            examples, tokenizer, 10, 5
        )

        self.assertIn("input_ids", example)
        self.assertIn("attention_mask", example)
        self.assertIn("target_ids", example)

        self.assertEqual(len(example["input_ids"]), 2)
        self.assertEqual(len(example["input_ids"][0]), 10)

        self.assertEqual(len(example["target_ids"]), 2)
        self.assertEqual(len(example["target_ids"][0]), 5)

    def test_multiple_dataloaders(self, tokenizer_mock):
        class _ConcreteClass(TextClassificationDataModule):
            @property
            def val_dataloader_names(self):
                return ["pt", "en"]

            @property
            def test_dataloader_names(self):
                return ["pt", "en", "es"]

            def prepare_datasets(self):
                train_data = create_mock_data()

                val_data_1 = create_mock_data()
                val_data_2 = create_mock_data()

                tst_data_1 = create_mock_data()
                tst_data_2 = create_mock_data()
                tst_data_3 = create_mock_data()

                val_data = DatasetDict({"pt": val_data_1, "en": val_data_2})
                tst_data = DatasetDict(
                    {"pt": tst_data_1, "en": tst_data_2, "es": tst_data_3}
                )

                return {"train": train_data, "validation": val_data, "test": tst_data}

        module = _ConcreteClass("pretrained", 10, 10, 10)
        module.setup()

        val_dataloaders = module.val_dataloader()
        tst_dataloaders = module.test_dataloader()

        self.assertIsInstance(val_dataloaders, list)
        self.assertEqual(len(val_dataloaders), 2)

        self.assertIsInstance(tst_dataloaders, list)
        self.assertEqual(len(tst_dataloaders), 3)


@patch("data_base.load_dataset")
@patch("data_base.AutoTokenizer.from_pretrained", return_value=FakeTokenizer())
class Assin2Test(TestCase):
    def test_load_datasets(self, _, load_dataset: Mock):
        load_dataset.return_value = [
            create_mock_data(label_column="entailment_judgment"),
            create_mock_data(label_column="entailment_judgment"),
            create_mock_data(label_column="entailment_judgment"),
        ]
        datasets = Assin2DataModule("pretrained", 10, 10, 32).prepare_datasets()

        load_dataset.assert_called_with(
            "assin2",
            None,
            split=["train", "validation", "test"],
            cache_dir=None,
            keep_in_memory=False,
        )

        self.assertIn("train", datasets)
        self.assertIn("validation", datasets)
        self.assertIn("test", datasets)

    def test_load_datasets_splits(self, _, load_dataset: Mock):
        load_dataset.return_value = [
            create_mock_data(label_column="entailment_judgment"),
            create_mock_data(label_column="entailment_judgment"),
            create_mock_data(label_column="entailment_judgment"),
        ]
        custom_splits = {"train": "train[:5%]", "validation": "validation_x"}
        expected_splits = ["train[:5%]", "validation_x", "test"]

        Assin2DataModule(
            "pretrained", 10, 10, 32, splits=custom_splits
        ).prepare_datasets()

        load_dataset.assert_called_with(
            "assin2", None, split=expected_splits, cache_dir=None, keep_in_memory=False
        )

    def test_dataset_columns(self, tokenizer, load_dataset):
        load_dataset.return_value = [
            create_mock_data(label_column="entailment_judgment"),
            create_mock_data(label_column="entailment_judgment"),
            create_mock_data(label_column="entailment_judgment"),
        ]

        datasets = Assin2DataModule("pretrained", 10, 10, 32).prepare_datasets()

        self.assertIn("label", list(datasets["train"].features.keys()))
        self.assertIn("label", list(datasets["validation"].features.keys()))
        self.assertIn("label", list(datasets["test"].features.keys()))


@patch("data_base.AutoTokenizer.from_pretrained", return_value=FakeTokenizer())
class XnliTest(TestCase):
    @patch("data_base.load_dataset", return_value=[Mock(), Mock(), Mock()])
    def test_load_datasets(self, load_dataset: Mock, _):
        load_dataset.return_value = [
            create_xnli_mock_data(),
            create_xnli_mock_data(),
            create_xnli_mock_data(),
        ]

        datasets = XnliDataModule("pretrained", 10, 10, 32).prepare_datasets()

        expected_lang_calls = [
            call(
                ANY,
                ANY,
                split=["train", "validation", "test"],
                cache_dir=None,
                keep_in_memory=False,
            )
        ]

        load_dataset.assert_has_calls(expected_lang_calls)

        self.assertIn("train", datasets)
        self.assertIn("validation", datasets)
        self.assertIn("test", datasets)

    @patch("data_base.load_dataset", return_value=[Mock(), Mock(), Mock()])
    def test_load_datasets_splits(self, load_dataset: Mock, _):
        custom_splits = {"train": "train[:5%]", "validation": "validation_x"}
        expected_lang_calls = [
            call(
                ANY,
                ANY,
                split=["train[:5%]", "validation_x", "test"],
                cache_dir=None,
                keep_in_memory=False,
            )
        ]

        load_dataset.return_value = [
            create_xnli_mock_data(),
            create_xnli_mock_data(),
            create_xnli_mock_data(),
        ]

        datasets = XnliDataModule(
            "pretrained", 10, 10, 32, splits=custom_splits
        ).prepare_datasets()

        load_dataset.assert_has_calls(expected_lang_calls)

    @unittest.skipUnless(RUN_SLOW == True, "Downloads the actual dataset")
    def test_return_multiple_dataloaders(self, _):
        module = XnliDataModule("pretrained", 10, 10, 32)
        module.setup("fit")

        val_dataloaders = module.val_dataloader()
        test_dataloaders = module.test_dataloader()

        self.assertIsInstance(val_dataloaders, list)
        self.assertIsInstance(val_dataloaders[0], DataLoader)
        self.assertEqual(len(val_dataloaders), len(module.XNLI_LANGUAGES))

        self.assertIsInstance(test_dataloaders, list)
        self.assertIsInstance(test_dataloaders[0], DataLoader)
        self.assertEqual(len(test_dataloaders), len(module.XNLI_LANGUAGES))

    def test_flatten(self, _):
        dataset_original = Dataset.from_dict(
            {
                "premise": [
                    {"en": "premise no 1", "pt": "premissa num 1"},
                    {"en": "premise no 2", "pt": "premissa num 2"},
                ],
                "hypothesis": [
                    {
                        "language": ["en", "pt"],
                        "translation": ["hypothesis no 1", "hipótese num 1"],
                    },
                    {
                        "language": ["en", "pt"],
                        "translation": ["hypothesis no 2", "hipótese num 2"],
                    },
                ],
                "label": [0, 1],
            }
        )

        expected_premise = [
            "premise no 1",
            "premissa num 1",
            "premise no 2",
            "premissa num 2",
        ]

        expected_hypothesis = [
            "hypothesis no 1",
            "hipótese num 1",
            "hypothesis no 2",
            "hipótese num 2",
        ]

        expected_labels = [0, 0, 1, 1]

        result = dataset_original.map(XnliDataModule.flatten, batched=True)

        self.assertListEqual(result["premise"], expected_premise)
        self.assertListEqual(result["hypothesis"], expected_hypothesis)
        self.assertListEqual(result["label"], expected_labels)

    def test_filter_by_lang(self, _):
        dataset = Dataset.from_dict(
            {
                "language": ["pt", "en", "fr", "pt", "es"],
                "texts": ["portuguese", "english", "french", "portuguese", "spanish"],
            }
        )

        pt_expected = ["portuguese", "portuguese"]
        en_expected = ["english"]
        es_expected = ["spanish"]
        fr_expected = ["french"]

        by_lang = DatasetDict(
            {
                lang: dataset.filter(
                    XnliDataModule.filter_data_by_lang, fn_kwargs={"language": lang}
                )
                for lang in ["pt", "en", "es", "fr"]
            }
        )

        self.assertListEqual(by_lang["pt"]["texts"], pt_expected)
        self.assertListEqual(by_lang["en"]["texts"], en_expected)
        self.assertListEqual(by_lang["es"]["texts"], es_expected)
        self.assertListEqual(by_lang["fr"]["texts"], fr_expected)
