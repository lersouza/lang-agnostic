import functools

import seqio
import tensorflow as tf

import preprocessors

DEFAULT_BYTE_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(vocabulary=seqio.ByteVocabulary(), required=False),
    "targets": seqio.Feature(vocabulary=seqio.ByteVocabulary()),
}

PRETRAIN_LANGUAGES = ("en",)


for lang in PRETRAIN_LANGUAGES:
    seqio.TaskRegistry.add(
        f"langagnostic.pretrain.{lang}.6B",
        source=seqio.TFExampleDataSource(
            split_to_filepattern={
                "train": f"gs://lang_agnostic/dataset/pretrain/mc4_{lang}_train_6000000000.tfrecord",
                "validation": f"gs://lang_agnostic/dataset/pretrain/mc4_{lang}_validation_315789473.tfrecord"
            },
            feature_description={
                "text": tf.io.FixedLenFeature([], tf.string, default_value=""),
            },
        ),
        preprocessors=[
            functools.partial(
                seqio.preprocessors.rekey, key_map={"inputs": None, "targets": "text"}
            ),
            seqio.preprocessors.tokenize,
            preprocessors.group_texts,
            seqio.preprocessors.append_eos_after_trim
        ],
        output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
        metric_fns=[],
    )
