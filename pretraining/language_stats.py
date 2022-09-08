from datasets import load_dataset
from tqdm.auto import tqdm

LANGUAGES = [
    "pt",
    "en",
    "de",
    "vi",
    "zh",
    "ru",
    "ar",
    "bn",
    "ko",
    "es",
]


with open("./language.stats", "w+", encoding="utf-8") as lang_stats_file:
    for language in tqdm(LANGUAGES):
        dataset = load_dataset("mc4", language, split="train", streaming=True)
        language_stats = {"examples": 0, "bytes": 0, "characters": 0, "bytes2char": 0}

        for example in dataset:
            example_in_bytes = list(example["text"].encode("utf-8"))
            
            language_stats["examples"] += 1
            language_stats["bytes"] += len(example_in_bytes)
            language_stats["characters"] += len(example["text"])
        
        language_stats["bytes2char"] = language_stats["bytes"] / language_stats["characters"]

        lang_stats_file.write("=" * 200)
        lang_stats_file.write(f"language: {language}")

        for key, value in language_stats.items():
            lang_stats_file.write(f"{key}: {value}\n")

        lang_stats_file.write("=" * 200)
        lang_stats_file.write("\n")
        
        