import re

from typing import Any, Dict, List, Tuple

from datasets import Dataset
from model_base import BaseSeq2SeqModelV2


class TextClassificationModel(BaseSeq2SeqModelV2):
    """
    A module for performing Text Classification Task (Natural Language Inference).
    """

    def post_process(
        self, examples: Dataset, features: Dataset, model_outputs: Dict[str, List[Any]]
    ) -> Dict[str, Any]:

        predicted_texts = self.tokenizer.batch_decode(
            model_outputs["predictions"], skip_special_tokens=True
        )

        predicted_labels = [self._convert_label_to_integer(t) for t in predicted_texts]
        target_labels = examples["label"]

        return {"predictions": predicted_labels, "references": target_labels}

    def _convert_label_to_integer(self, text: str):
        return int(text.strip()) if re.match(r"^\d+$", text.strip()) else -1

    def format_outputs_for_logging(self, predictions, references) -> Tuple:
        """
        Returns the model's predictions and references for additional logging.
        """
        return ["predictions", "references"], [
            [str(p), str(r)] for p, r in zip(predictions, references)
        ]
