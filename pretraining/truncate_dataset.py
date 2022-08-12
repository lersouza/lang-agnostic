"""
This script truncates mc4's language subset to a specified number of Byte-level tokens.
"""
import argparse

from pathlib import Path
from typing import Any, Counter, Dict

import tensorflow as tf

from seqio.vocabularies import ByteVocabulary
from tqdm.auto import tqdm
from datasets import load_dataset


#  The default max tokens are based on our pretraining intention:
#  - Train for 1M steps with batches of 2^16 tokens
#  - We add an extra 20% since ByT5 scripts will try to group tokens from
#    different examples to reach the optimum batch size regarding # of tokens
DEFAULT_MAX_TOKENS = int(2**16 * 1_000_000 * 1.2)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def save_stats(target_file_name: Path, stats: Dict[str, Any]):
    """
    Save a stats file for the `target_file_name`, using the `stats` value provided.
    """
    with open(str(target_file_name) + ".stats", "w+", encoding="utf-8") as stats_file:
        for key, value in stats.items():
            stats_file.write(f"{key}: {value}\n")


def print_stats(
    stats: Dict[str, Any], additional_info: Dict[str, Any] = None, complete: bool = True
):
    """Print statistics for followup"""
    status_message = "Done truncating." if complete else "Intermediary Stats."

    print("=" * 100)
    print(status_message)
    print("Stats:", stats)

    if additional_info:
        for info_name, info_value in additional_info.items():
            print(info_name, ":", info_value)

    print("=" * 100)


def truncate(
    language: str,
    split: str,
    max_train_tokens: int,
    validation_percentage: float,
    output_directory: Path,
):
    """
    Truncate the specified mC4's `language` subset to a maximum of `max_tokens`
    based on the result of `seqio.vocabularies.ByteVocabulary.encode`

    The output is saved as a TFRecord File to `output_directory`.
    A stats file is also saved in this directory.
    """
    tokens_to_process = (
        max_train_tokens
        if split == "train"
        else int(
            (validation_percentage * max_train_tokens) / (1.0 - validation_percentage)
        )
    )
    target_file_name = output_directory / f"mc4_{language}_{split}_{tokens_to_process}.tfrecord"

    original_dataset = load_dataset("mc4", language, split=split, streaming=True)
    vocabulary = ByteVocabulary()  # No special tokens are added for ByT5
    processed_urls = []

    stats = {
        "language": language,
        "split": split,
        "examples": 0,
        "original_text_length": 0,
        "text_length_after_truncation": 0,
        "tokens": 0,
        "original_tokens_length": 0,
        "max_tokens": tokens_to_process,
        "max_train_tokens": max_train_tokens,
        "validation_percentage": validation_percentage,
        "token2text_rate": None,
        "dropped_text_length": 0,
        "dropped_tokens_length": 0,
    }

    with tf.io.TFRecordWriter(str(target_file_name)) as file_writer, tqdm(
        total=tokens_to_process
    ) as pbar:
        for example in original_dataset:
            raw_text = example["text"]
            in_bytes = vocabulary.encode(raw_text)

            processed_urls.append(example["url"])

            stats["original_text_length"] += len(raw_text)
            stats["original_tokens_length"] += len(in_bytes)

            if len(in_bytes) + stats["tokens"] > tokens_to_process:
                remaining = int(tokens_to_process - stats["tokens"])

                # Truncate at the UTF-8 Byte level here
                # The main issue is that we may loose valid characters, since UTF-8 can take up to
                # 4 bytes for representing a single character and we may truncate in the middle.
                # However, since the ByteVocabulary has a fallback mechanism,
                # we may not have errors, just less characters in the end and model that
                # sees invalid bytes at the end of an epoch (if it comes to that, anyway)
                #
                # TODO Come up with a better way for that
                in_bytes = in_bytes[:remaining]
                raw_text = vocabulary.decode(in_bytes)

            file_writer.write(
                tf.train.Example(
                    features=tf.train.Features(
                        feature={"text": _bytes_feature(raw_text.encode("utf-8"))}
                    )
                ).SerializeToString()
            )

            stats["text_length_after_truncation"] += len(raw_text)
            stats["examples"] += 1
            stats["tokens"] += len(in_bytes)

            pbar.update(len(in_bytes))

            if stats["tokens"] >= tokens_to_process:
                break

    stats["token2text_rate"] = stats["tokens"] / stats["text_length_after_truncation"]
    stats["dropped_tokens_length"] = stats["original_tokens_length"] - stats["tokens"]
    stats["dropped_text_length"] = (
        stats["original_text_length"] - stats["text_length_after_truncation"]
    )

    save_stats(target_file_name, stats)
    print_stats(
        stats,
        additional_info={
            "Top 3 Processed URLS": Counter(processed_urls).most_common(n=3)
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("language", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--max_train_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=(
            "The maximum number of tokens to include in the final train dataset."
            f" Default is {DEFAULT_MAX_TOKENS}."
        ),
    )
    parser.add_argument(
        "--validation_percentage",
        type=float,
        default=0.05,
        help=(
            "The percentage of tokens that a validation split represent from the total"
            " (train + val) dataset. This only applies if split == validation"
        ),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("./"))

    args = parser.parse_args()

    truncate(
        args.language,
        args.split,
        args.max_train_tokens,
        args.validation_percentage,
        args.output_dir,
    )
