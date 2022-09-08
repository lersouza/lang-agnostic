import functools

import seqio
import t5
import tensorflow as tf

DEFAULT_BYTE_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(vocabulary=seqio.ByteVocabulary()),
    "targets": seqio.Feature(vocabulary=seqio.ByteVocabulary()),
}

PRETRAIN_LANGUAGES = ("en",)
MAX_PRETRAIN_TOKENS = int(2**16 * 1_000_000 * 1.2)
MAX_VALIDATION_TOKENS = int((MAX_PRETRAIN_TOKENS * 0.05) / 0.95)
MEAN_NOISE_SPAN_LENGTH = 20


for lang in PRETRAIN_LANGUAGES:
    seqio.TaskRegistry.add(
        f"monobyte.pretrain.{lang}",
        source=seqio.TFExampleDataSource(
            split_to_filepattern={
                "train": f"gs://monobyte/dataset/mc4_{lang}_train_{MAX_PRETRAIN_TOKENS}.tfrecord",
                "validation": f"gs://monobyte/dataset/mc4_{lang}_validation_{MAX_VALIDATION_TOKENS}.tfrecord"
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
            functools.partial(
                t5.data.preprocessors.span_corruption,
                mean_noise_span_length=MEAN_NOISE_SPAN_LENGTH,
            ),
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
        metric_fns=[],
    )
