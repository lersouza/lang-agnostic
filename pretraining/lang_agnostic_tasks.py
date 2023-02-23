import functools

import seqio
import tensorflow as tf

DEFAULT_BYTE_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(vocabulary=seqio.ByteVocabulary()),
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
        ],
        output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
        metric_fns=[],
    )
