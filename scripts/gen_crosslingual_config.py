""" Generates multiple config files for experiments with cross lingua transfer. """

import argparse
import functools
import itertools
import logging
import os
import re

from pathlib import Path
from typing import Dict, Any, List, Tuple


LOGGER = logging.getLogger("xlanggen")

MODEL_MAP = {
    "pt": ("hugo/byt5-mono-pt-v1", "e3ce3d12e1516acc756cb0f5b124126789d5f3f7"),
    "en": ("hugo/byt5-mono-en-v1", "939ad92a33a0f1fbb951d7a93b339be5bc3500b2"),
    "de": ("hugo/byt5-mono-de-v1", "0b2e0002dcf0586ac8d65e417be1506241cc92d5"),
    "vi": ("hugo/byt5-mono-vi-v1", "4438ba79c83fc7387ebf67256cefefe26171633b"),
    "zh": ("hugo/byt5-mono-zh-v1", "ccb30bb94be2a65077483371672d1d04388d77d9"),
    "ru": ("hugo/byt5-mono-ru-v1", "72dd45751d10e64ddae03841106488c8fa37eafb"),
    "ar": ("hugo/byt5-mono-ar-v1", "73b0b3095f36e3369842bfa33659e326a7aa661e"),
    "bn": ("hugo/byt5-mono-bn-v1", "273dafaf63f799a4d6865aa482e1acaba83fd8dd"),
    # "sw": ("hugo/byt5-mono-sw-v1", "b3c60b74c853d988ac1c425bc174456307f8f1bc"),
    "ko": ("hugo/byt5-mono-ko-v1", "984cca6baf64e3f2b81d44632d2fd0a42f8be087"),
    "es": ("hugo/byt5-mono-es-v1", "19f1f4683914f6e337718f249ceb62acb0e2bd1e"),
    "hierarchical": ("hugo/byt5-mono-hierarchical-v1", "5bd9ddc9d25cde834842e2dae992601ab2424446"),
    "nonsense": ("hugo/byt5-mono-nonsense-v1", "5e4676baa37e04ae9d74132bd2b73b39027d193d"),
    "multi": ("google/byt5-small", "null"),
}

BASE_DIR = Path(os.path.dirname(os.path.realpath(__file__))).parent
BASELINES = ["hierarchical", "nonsense", "multi"]
AVAILABLE_SOURCE_LANGUAGES = set(MODEL_MAP.keys()) - set(BASELINES)

PRESETS = {
    "tydi_qa": {
        "template": BASE_DIR / "configs/templates/standard-tydi-qa.yaml",
        "output_dir": BASE_DIR / "configs/foreign_lang_exps/",
        "file_prefix": "qa",
        "source_languages": ["ar", "bn", "en", "ko", "ru"],
        "target_languages": ["ar", "bn", "en", "ko", "ru"],
        "override": False,
    },
    "sberquad": {
        "template": BASE_DIR / "configs/templates/sberquad.yaml",
        "output_dir": BASE_DIR / "configs/sberquad_experiments/",
        "file_prefix": "qa",
        "source_languages": AVAILABLE_SOURCE_LANGUAGES,
        "target_languages": ["ru"],
        "override": False,
    },
    "squad": {
        "template": BASE_DIR / "configs/templates/squad.yaml",
        "output_dir": BASE_DIR / "configs/squad_experiments/",
        "file_prefix": "qa",
        "source_languages": AVAILABLE_SOURCE_LANGUAGES,
        "target_languages": ["en"],
        "override": False,
    },
    "korquad": {
        "template": BASE_DIR / "configs/templates/korquad.yaml",
        "output_dir": BASE_DIR / "configs/korquad_experiments/",
        "file_prefix": "qa",
        "source_languages": AVAILABLE_SOURCE_LANGUAGES,
        "target_languages": ["ko"],
        "override": False,
    },
    "korquad-baselines": {
        "template": BASE_DIR / "configs/templates/korquad.yaml",
        "output_dir": BASE_DIR / "configs/korquad_baselines/",
        "file_prefix": "qa",
        "source_languages": BASELINES,
        "target_languages": ["ko"],
        "override": False,
    },
    "sberquad-baselines": {
        "template": BASE_DIR / "configs/templates/sberquad.yaml",
        "output_dir": BASE_DIR / "configs/sberquad_baselines/",
        "file_prefix": "qa",
        "source_languages": BASELINES,
        "target_languages": ["ru"],
        "override": False,
    },
}


def replace(match: re.Match, context: Dict[str, Any]):
    """
    Replaces the `match`ed variable string with the corresponding variable from `context`.
    """

    variable_name = match.group(1)
    variable_value = context.get(variable_name)

    return variable_value


def process_pair(lang_pair: Tuple, template_file: str) -> List[str]:
    """
    Generates a file content (as a list of lines) for a pair of languages (source, target).
    """
    source, target = lang_pair
    model_name, model_hash = MODEL_MAP.get(source)

    repl_context = {
        "SOURCE_LANGUAGE": source,
        "TARGET_LANGUAGE": target,
        "MODEL_NAME": model_name,
        "MODEL_CHECKPOINT_HASH": model_hash,
    }

    repl_partial = functools.partial(replace, context=repl_context)
    new_file_def = []

    with open(template_file, "r", encoding="utf-8") as template:
        for template_line in template:
            new_file_def.append(re.sub(r"\${(\w*)}", repl_partial, template_line))

    return new_file_def


def main(args):
    """Main processing method."""
    user_selection = PRESETS.get(args.preset_name)

    source = user_selection["source_languages"]
    target = user_selection["target_languages"]

    all_lang_pairs = itertools.product(source, target)

    # Ensure output dir is created
    os.makedirs(user_selection["output_dir"], exist_ok=True)

    for pair in all_lang_pairs:
        config_content = process_pair(pair, user_selection["template"])

        file_name = f"{user_selection['file_prefix']}-{pair[0]}-{pair[1]}.yaml".format(
            user_selection["file_prefix"], *pair
        )
        file_path = user_selection["output_dir"] / file_name

        if file_path.exists() and not args.override:
            LOGGER.info("File %s already exists.", file_path)
            continue

        with open(file_path, "w+", encoding="utf-8") as tgt:
            tgt.writelines(config_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("preset_name")
    parser.add_argument("--override", action="store_true")

    parsed_args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(parsed_args)
