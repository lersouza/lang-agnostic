# pylint: disable=abstract-method

import re

import numpy as np
import torch

from typing import Any, Dict, List, Tuple, Union
from model_base import BaseSeq2SeqModel


class QuestionAnsweringModel(BaseSeq2SeqModel):
    """
    A module for performing Question-Answering tasks.
    """

    def format_model_predictions(
        self,
        dataloader_index: int,
        dataloader_name: str,
        batch: Dict[str, Union[torch.Tensor, Any]],
        output: torch.Tensor,
    ) -> List[Any]:
        """
        Convert model's output tokens to a string containing the predicted answer.
        """
        predicted_texts = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        examples_ids = batch["id"]

        return [
            {"id": i, "predicted_text": t}
            for i, t in zip(examples_ids, predicted_texts)
        ]

    def format_batch_references(
        self,
        dataloader_index: int,
        dataloader_name: str,
        batch: Dict[str, Union[torch.Tensor, Any]],
    ):
        """
        Returns the reference answers for the batch.

        Parameters:
            dataloader_index (int):
                The index of the dataloader being processed at this time

            dataloader_name (str):
                The name of the dataloader being processed at this time

            batch (Dict[Union[Tensor, Any]]):
                The batch for which predictions were made.
        """
        return [
            {"id": i, "answers": a}
            for i, a in zip(batch["id"], batch["answers"])
        ]


    def format_outputs_for_logging(self, predictions, references) -> Tuple:
        """
        Returns the model's predictions and references for additional logging.
        """
        return ["predictions", "references"], [
            [str(p), str(r)] for p, r in zip(predictions, references)
        ]
