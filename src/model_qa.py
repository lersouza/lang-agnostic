""" A module for traning and evaluating in Question Answering Tasks. """
# pylint: disable=abstract-method

from typing import Any, Dict, List, Tuple

from datasets import Dataset
from model_base import BaseSeq2SeqModelV2


class QuestionAnsweringModel(BaseSeq2SeqModelV2):
    """
    A module for performing Question-Answering tasks.
    """

    def post_process(
        self, examples: Dataset, features: Dataset, model_outputs: Dict[str, List[Any]]
    ) -> Dict[str, Any]:

        predictions, references = [], []
        decoded_predictions = self.tokenizer.batch_decode(
            model_outputs["predictions"], skip_special_tokens=True
        )

        for example, prediction in zip(examples, decoded_predictions):
            predictions.append({"id": example["id"], "prediction_text": prediction})
            references.append({"id": example["id"], "answers": example["answers"]})

        return {"predictions": predictions, "references": references}

    def format_outputs_for_logging(self, predictions, references) -> Tuple:
        """
        Returns the model's predictions and references for additional logging.
        """
        return ["predictions", "references"], [
            [p["prediction_text"], r["answers"]]
            for p, r in zip(predictions, references)
        ]
