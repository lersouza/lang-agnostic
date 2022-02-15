import re

import numpy as np
import torch

from typing import Any, Dict, List, Tuple, Union
from model_base import BaseSeq2SeqModel


class TextClassificationModel(BaseSeq2SeqModel):
    """
    A module for performing Text Classification Task (Natural Language Inference).
    """

    def format_model_predictions(
        self,
        dataloader_index: int,
        dataloader_name: str,
        batch: Dict[str, Union[torch.Tensor, Any]],
        output: torch.Tensor,
    ) -> List[Any]:
        """
        Convert model's output tokens to an integer representation of a predicted label.

        First, the `output` is decoded using the model's tokenizer.
        Then, every value is casted as integer.

        If an invalid integer label is produced (a string "invalid", for instance),
        we assume `-1` as the output, since this corresponds to an invalid label.
        """
        s2i = lambda text: int(text.strip()) if re.match(r"^\d+$", text.strip()) else -1

        prediction_texts = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        predicted_labels = [s2i(t) for t in prediction_texts]

        return predicted_labels

    def format_batch_references(
        self,
        dataloader_index: int,
        dataloader_name: str,
        batch: Dict[str, Union[torch.Tensor, Any]],
    ):
        """
        Returns the expected labels from the batch.

        Parameters:
            dataloader_index (int):
                The index of the dataloader being processed at this time

            dataloader_name (str):
                The name of the dataloader being processed at this time

            batch (Dict[Union[Tensor, Any]]):
                The batch for which predictions were made.
        """
        return batch["labels"].tolist()

    def format_outputs_for_logging(self, predictions, references) -> Tuple:
        """
        Returns the model's predictions and references for additional logging.
        """
        return ["predictions", "references"], [
            [str(p), str(r)] for p, r in zip(predictions, references)
        ]
