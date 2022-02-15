import unittest
import torch

from argparse import Namespace
from unittest.mock import Mock, patch

from model_base import BaseSeq2SeqModel


class FakeModelForTesting(BaseSeq2SeqModel):
    def __init__(
        self,
        pretrained_model_name: str = None,
        pretrained_model_revision: str = None,
        use_pretrained_weights: bool = True,
        max_target_length: int = 128,
        metric_name: str = "accuracy",
        from_flax: bool = False,
        val_dataloader_names=None,
        test_datalodader_names=None,
    ):
        self.forward_calls = []
        self.log_calls = []
        self.training = True
        self._val_dl_names = val_dataloader_names or ["val_default"]
        self._test_dl_names = test_datalodader_names or ["test_default"]

    @property
    def val_dataloader_names(self):
        return self._val_dl_names

    @property
    def test_dataloader_names(self):
        return self._test_dl_names

    def forward(self, *args, **kwargs):
        self.forward_calls.append(kwargs)

        if self.training:
            return Namespace(loss=0.005)
        else:
            return torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    def log(self, name: str, value, **kwargs):
        self.log_calls.append({"name": name, "value": value, **kwargs})

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class BaseModelTestCase(unittest.TestCase):
    @patch("model_base.load_metric")
    @patch("model_base.AutoConfig.from_pretrained")
    @patch("model_base.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("model_base.AutoTokenizer.from_pretrained")
    def test_initialization_default(
        self,
        auto_tokenizer: Mock,
        auto_model: Mock,
        auto_config: Mock,
        load_metric: Mock,
    ):
        config_mock = Mock()
        auto_config.return_value = config_mock

        model = BaseSeq2SeqModel("pretrained_name")

        auto_tokenizer.assert_called_once_with("pretrained_name")
        auto_model.assert_called_once_with(
            "pretrained_name", config=config_mock, revision=None, from_flax=False
        )
        auto_config.assert_called_once_with("pretrained_name")
        load_metric.assert_called_once_with("accuracy")

    @patch("model_base.load_metric")
    @patch("model_base.AutoConfig.from_pretrained")
    @patch("model_base.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("model_base.AutoTokenizer.from_pretrained")
    def test_initialization_with_values(
        self,
        auto_tokenizer: Mock,
        auto_model: Mock,
        auto_config: Mock,
        load_metric: Mock,
    ):
        config_mock = Mock()
        auto_config.return_value = config_mock

        model = BaseSeq2SeqModel(
            pretrained_model_name="pretrained_name",
            pretrained_model_revision="my_revision",
            use_pretrained_weights=True,
            max_target_length=64,
            metric_name="squad",
            from_flax=True,
        )

        auto_tokenizer.assert_called_once_with("pretrained_name")
        auto_model.assert_called_once_with(
            "pretrained_name",
            config=config_mock,
            revision="my_revision",
            from_flax=True,
        )
        auto_config.assert_called_once_with("pretrained_name")
        load_metric.assert_called_once_with("squad")

        self.assertEqual(model.hparams.max_target_length, 64)

    @patch("model_base.load_metric")
    @patch("model_base.AutoConfig.from_pretrained")
    @patch("model_base.AutoModelForSeq2SeqLM.from_config")
    @patch("model_base.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("model_base.AutoTokenizer.from_pretrained")
    def test_initialization_without_pretraining(
        self,
        auto_tokenizer: Mock,
        auto_model: Mock,
        auto_model_from_config: Mock,
        auto_config: Mock,
        load_metric: Mock,
    ):
        config_mock = Mock()
        auto_config.return_value = config_mock

        model = BaseSeq2SeqModel(
            pretrained_model_name="pretrained_name",
            pretrained_model_revision="my_revision",
            use_pretrained_weights=False,
            max_target_length=64,
            metric_name="squad",
            from_flax=True,
        )

        auto_tokenizer.assert_called_once_with("pretrained_name")
        auto_config.assert_called_once_with("pretrained_name")
        auto_model.assert_not_called()
        auto_model_from_config.assert_called_once_with(config_mock)
        load_metric.assert_called_once_with("squad")

        self.assertEqual(model.hparams.max_target_length, 64)

    def test_train_step(self):
        model = FakeModelForTesting()

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]),
            "attention_mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
            "target_ids": [[4], [5]],
            "label": [1, 1, 1, 1],
        }

        model.training_step(batch, 0)

        self.assertListEqual(model.forward_calls, [batch])
        self.assertListEqual(
            model.log_calls,
            [
                {"name": "train/loss", "value": 0.005},
                {"name": "train/seq_len", "value": 4.0},
            ],
        )

    def test_valid_step(self):
        model = FakeModelForTesting()
        model.training = False
        model.format_model_predictions = Mock(return_value=["predictions"])
        model.format_batch_references = Mock(return_value=["refs"])

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]),
            "attention_mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
            "target_ids": [[4], [5]],
            "label": [1, 1, 1, 1],
        }

        result = model.validation_step(batch, 0, 0)

        self.assertEqual(
            result, {"predictions": ["predictions"], "references": ["refs"]}
        )
        self.assertListEqual(model.forward_calls, [batch])
        self.assertListEqual(model.log_calls, [{"name": "val/seq_len", "value": 4.0}])

    def test_multi_dataloader_aggregate_outputs(self):
        dataloader_0_outs = [{"a": [0, 1, 2]}, {"a": [2, 3, 4]}]
        dataloader_1_outs = [{"a": [4, 5, 6]}, {"a": [7, 8, 9]}]

        dataloader_names = ["dataloader_0", "dataloader_1"]
        val_outputs = [dataloader_0_outs, dataloader_1_outs]

        expected = {
            "dataloader_0": {"a": [0, 1, 2, 2, 3, 4]},
            "dataloader_1": {"a": [4, 5, 6, 7, 8, 9]},
        }
        actual = BaseSeq2SeqModel.aggregate_outputs(val_outputs, dataloader_names)

        self.assertEqual(expected, actual)

    def test_single_dataloader_aggregate_outputs(self):
        dataloader_0_outs = [{"a": [0, 1, 2]}, {"a": [2, 3, 4]}]
        dataloader_names = ["dataloader_0"]

        expected = {
            "dataloader_0": {"a": [0, 1, 2, 2, 3, 4]},
        }
        actual = BaseSeq2SeqModel.aggregate_outputs(dataloader_0_outs, dataloader_names)

        self.assertEqual(expected, actual)
