from dataclasses import dataclass


@dataclass
class DataDef:
    dataset_name: str
    premise_column: str
    hypothesis_column: str
    label_column: str
    num_labels: int


def data_def_name(dataset_name, subdataset_name):
    return f"{dataset_name}:{subdataset_name or 'all'}"


DATA_DEFS = {
    "glue:mnli": DataDef("mnli", "premise", "hypothesis", "label", 3),
    "assin2:all": DataDef("assin2", "premise", "hypothesis", "entailment_judgment", 2),
}


def extract_seq2seq_features(
    dataset_definition, tokenizer, max_length, target_max_length, dataset
):
    hypothesis = dataset_definition.hypothesis_column
    premise = dataset_definition.premise_column
    label_column = dataset_definition.label_column

    def preprocess_sample(example):
        sentence = f"premise: {example[premise]}. hypothesis: {example[hypothesis]}."
        original_label = example[label_column]

        encoded = tokenizer(
            sentence,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=False,
        )

        target_encoded = tokenizer(
            str(original_label),
            max_length=target_max_length,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=False,
        )

        encoded["target_ids"] = target_encoded["input_ids"]
        encoded["label"] = original_label

        return encoded

    features = dataset.map(
        preprocess_sample, batched=False, remove_columns=[hypothesis, premise]
    )

    features.set_format(type="torch")

    return features
