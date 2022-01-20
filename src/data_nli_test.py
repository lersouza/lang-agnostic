from typing import Dict
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch, call, ANY

from datasets import Dataset

from data_nli import Assin2DataModule, TextClassificationDataModule, XnliDataModule
from test_utils import FakeTokenizer


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


@patch("data_nli.AutoTokenizer.from_pretrained", return_value=FakeTokenizer())
class TextClassificationTest(TestCase):
    def test_init(self, tokenizer_mock: Mock):
        module = TextClassificationDataModule(
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
        module = TextClassificationDataModule("pretrained", 10, 10, 32)
        module.prepare_datasets = Mock()

        module.prepare_data()

        module.prepare_datasets.assert_called_once()

    def test_setup(self, tokenizer_mock):
        module = TextClassificationDataModule("pretrained", 10, 10, 32)

        module.prepare_datasets = Mock(
            return_value={"train": ["train"], "validation": ["val"]}
        )
        module.preprocess = Mock(side_effect=[["features_train"], ["features_val"]])
        module.prepare_features_for_model = Mock()

        module.setup("fit")

        module.prepare_datasets.assert_called_once()
        module.preprocess.assert_has_calls(
            [call(["train"], "train"), call(["val"], "validation")]
        )
        module.prepare_features_for_model.assert_has_calls(
            [call("train"), call("validation")]
        )

        self.assertEqual(module.data["train"], ["train"])
        self.assertEqual(module.data["validation"], ["val"])

        self.assertEqual(module.features["train"], ["features_train"])
        self.assertEqual(module.features["validation"], ["features_val"])

    @patch("data_nli.DataLoader")
    def test_train_dataloader(self, dataloader_mock: Mock, tokenizer_mock):
        module = TextClassificationDataModule(
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

    @patch("data_nli.DataLoader")
    def test_val_dataloader(self, dataloader_mock: Mock, *args):
        module = TextClassificationDataModule(
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

    def test_preprocess(self, tokenizer_mock: Mock):
        module = TextClassificationDataModule("pretrained", 10, 10, 32)
        dataset = create_mock_data()

        result = module.preprocess(dataset, "train")

        tokenizer_mock.assert_called()

        self.assertIsInstance(result, Dataset)
        self.assertEqual(len(result), len(dataset))

    def test_input_sequences(self, tokenizer_mock):
        example = {"premise": "my premise", "hypothesis": "my hypothesis", "label": 2}

        expected_input = "premise: my premise. hypothesis: my hypothesis."
        expected_target = "2"

        example = TextClassificationDataModule.prepare_input_sentence(example)

        self.assertIn("input", example)
        self.assertIn("target", example)

        self.assertEqual(example["input"], expected_input)
        self.assertEqual(example["target"], expected_target)


@patch("data_nli.load_dataset")
@patch("data_nli.AutoTokenizer.from_pretrained", return_value=FakeTokenizer())
class Assin2Test(TestCase):
    def test_load_datasets(self, _, load_dataset: Mock):
        datasets = Assin2DataModule("pretrained", 10, 10, 32).prepare_datasets()

        load_dataset.assert_called_with("assin2", split=["train", "validation"])

        self.assertIn("train", datasets)
        self.assertIn("validation", datasets)

    def test_load_datasets_splits(self, _, load_dataset: Mock):
        custom_splits = {"train": "train[:5%]", "validation": "validation_x"}
        expected_splits = ["train[:5%]", "validation_x"]

        Assin2DataModule(
            "pretrained", 10, 10, 32, splits=custom_splits
        ).prepare_datasets()

        load_dataset.assert_called_with("assin2", split=expected_splits)

    def test_dataset_columns(self, tokenizer, load_dataset):
        train_mock, valid_mock = MagicMock(), MagicMock()
        load_dataset.return_value = [train_mock, valid_mock]

        datasets: Dict[str, Mock] = Assin2DataModule(
            "pretrained", 10, 10, 32
        ).prepare_datasets()

        train_mock.rename_column.assert_called_with("entailment_judgment", "label")
        valid_mock.rename_column.assert_called_with("entailment_judgment", "label")


@patch("data_nli.load_dataset")
@patch("data_nli.AutoTokenizer.from_pretrained", return_value=FakeTokenizer())
class XnliTest(TestCase):
    def test_load_datasets(self, _, load_dataset: Mock):
        datasets = XnliDataModule("pretrained", 10, 10, 32).prepare_datasets()

        load_dataset.assert_has_calls(
            [
                call("xnli", "en", split="train"),
                call("xnli", "all_languages", split="validation"),
            ]
        )

        self.assertIn("train", datasets)
        self.assertIn("validation", datasets)

    def test_load_datasets_splits(self, _, load_dataset: Mock):
        custom_splits = {"train": "train[:5%]", "validation": "validation_x"}

        datasets = XnliDataModule(
            "pretrained", 10, 10, 32, splits=custom_splits
        ).prepare_datasets()

        load_dataset.assert_has_calls(
            [
                call("xnli", "en", split="train[:5%]"),
                call("xnli", "all_languages", split="validation_x"),
            ]
        )

    def test_flatten_when_necessary(self, _, load_dataset: Mock):
        languages = {"train": "en", "validation": "all_languages"}

        train_mock, valid_mock = MagicMock(), MagicMock()
        load_dataset.side_effect = [train_mock, valid_mock]

        module = XnliDataModule("pretrained", 10, 10, 32, languages=languages)
        module.prepare_datasets()

        train_mock.map.assert_not_called()
        valid_mock.map.assert_called_with(module.flatten, batched=True, desc=ANY)
