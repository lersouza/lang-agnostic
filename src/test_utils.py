from typing import Any

from transformers import PreTrainedTokenizer


class FakeTokenizer(PreTrainedTokenizer):
    @property
    def is_fast(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        return 30000

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        max_length = kwds.get("max_length", 10)
        samples_length = 1 if isinstance(args[0], str) else len(args[0])

        input_ids = list(range(max_length))
        attention_mask = [1] * max_length

        return {
            "input_ids": [input_ids] * samples_length,
            "attention_mask": [attention_mask] * samples_length,
        }