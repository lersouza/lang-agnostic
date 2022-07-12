"""TODO(tydiqa): Add a description here."""


import json
import textwrap

import datasets
from datasets.tasks import QuestionAnsweringExtractive


# TODO(tydiqa): BibTeX citation
_CITATION = """\
@article{tydiqa,
title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
year    = {2020},
journal = {Transactions of the Association for Computational Linguistics}
}
"""

# TODO(tydiqa):
_DESCRIPTION = """\
TyDi QA is a question answering dataset covering 11 typologically diverse languages with 204K question-answer pairs.
The languages of TyDi QA are diverse with regard to their typology -- the set of linguistic features that each language
expresses -- such that we expect models performing well on this set to generalize across a large number of the languages
in the world. It contains language phenomena that would not be found in English-only corpora. To provide a realistic
information-seeking task and avoid priming effects, questions are written by people who want to know the answer, but
don’t know the answer yet, (unlike SQuAD and its descendents) and the data is collected directly in each language without
the use of translation (unlike MLQA and XQuAD).
"""

_URL = "https://storage.googleapis.com/tydiqa/"

_GOLDP_URLS = {
    "train": _URL + "v1.1/tydiqa-goldp-v1.1-train.json",
    "dev": _URL + "v1.1/tydiqa-goldp-v1.1-dev.json",
}


class TydiqaGoldPConfig(datasets.BuilderConfig):

    """BuilderConfig for Tydiqa"""

    GOLDP_DESCRIPTION = textwrap.dedent(
        """
        Gold passage task (GoldP) for {lang} language(s): Given a passage that is guaranteed
        to contain the answer, predict the single contiguous span of characters that answers the
        question. This is more similar to existing reading comprehension datasets (as opposed to
        the information-seeking task outlined above).

        This task is constructed with two goals in mind: (1) more directly comparing with prior work
        and (2) providing a simplified way for researchers to use TyDi QA by providing compatibility
        with existing code for SQuAD 1.1, XQuAD, and MLQA. Toward these goals, the gold passage
        task differs from the primary task in several ways: only the gold answer passage is
        provided rather than the entire Wikipedia article; unanswerable questions have been
        discarded, similar to MLQA and XQuAD; we evaluate with the SQuAD 1.1 metrics like XQuAD;
        and Thai and Japanese are removed since the lack of whitespace breaks some tools.
          """
    )

    def __init__(self, **kwargs):
        """

        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(TydiqaGoldPConfig, self).__init__(
            version=datasets.Version("1.0.0", ""), **kwargs
        )

        self.description = self.GOLDP_DESCRIPTION.format(lang=self.name)


class TydiqaGoldP(datasets.GeneratorBasedBuilder):
    """ TydiQA-GoldP dataset with language option"""

    LANGUAGES = [
        "english",
        "arabic",
        "bengali",
        "finnish",
        "indonesian",
        "swahili",
        "korean",
        "russian",
        "telugu",
    ]

    BUILDER_CONFIGS = [TydiqaGoldPConfig(name=lang) for lang in LANGUAGES] + [
        TydiqaGoldPConfig(name="all")
    ]

    def _info(self):

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://github.com/google-research-datasets/tydiqa",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question",
                    context_column="context",
                    answers_column="answers",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        goldp_downloaded = dl_manager.download_and_extract(_GOLDP_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": goldp_downloaded["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": goldp_downloaded["dev"]},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for article in data["data"]:
                title = article.get("title", "").strip()
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]

                        # Make available only examples in the selected language
                        if self.config.name != "all" and not id_.startswith(
                            self.config.name
                        ):
                            continue

                        answer_starts = [
                            answer["answer_start"] for answer in qa["answers"]
                        ]
                        answers = [answer["text"].strip() for answer in qa["answers"]]

                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield id_, {
                            "title": title,
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
