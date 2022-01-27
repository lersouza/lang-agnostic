import unittest

from model_nli import aggregate_outputs


class GenericModelFunctionTest(unittest.TestCase):
    def test_multi_dataloader_aggregate_outputs(self):
        dataloader_0_outs = [{"a": [0, 1, 2]}, {"a": [2, 3, 4]}]
        dataloader_1_outs = [{"a": [4, 5, 6]}, {"a": [7, 8, 9]}]

        dataloader_names = ["dataloader_0", "dataloader_1"]
        val_outputs = [dataloader_0_outs, dataloader_1_outs]

        expected = {
            "dataloader_0": {"a": [0, 1, 2, 2, 3, 4]},
            "dataloader_1": {"a": [4, 5, 6, 7, 8, 9]},
        }
        actual = aggregate_outputs(val_outputs, dataloader_names)

        self.assertEqual(expected, actual)

    def test_single_dataloader_aggregate_outputs(self):
        dataloader_0_outs = [{"a": [0, 1, 2]}, {"a": [2, 3, 4]}]
        dataloader_names = ["dataloader_0"]

        expected = {
            "dataloader_0": {"a": [0, 1, 2, 2, 3, 4]},
        }
        actual = aggregate_outputs(dataloader_0_outs, dataloader_names)

        self.assertEqual(expected, actual)
