"""SQUAD: The Stanford Question Answering Dataset."""


import json
import datasets

from datasets.tasks import QuestionAnsweringExtractive


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{Sayama2019,
   abstract = {Academic secretaries and faculty members of higher education institutions face a common problem: the abundance of questions sent by academics whose answers are found in available institutional documents. The official documents produced by Brazilian public universities are vast and disperse, which discourage students to further search for answers in such sources. In order to lessen this problem, we present FaQuAD: a novel machine reading comprehension dataset in the domain of Brazilian higher education institutions. FaQuAD follows the format of SQuAD (Stanford Question Answering Dataset) [Rajpurkar et al.2016]. It comprises 900 questions about 249 reading passages(paragraphs), which were taken from 18 official documents of a computer science college from a Brazilian federal university and 21 Wikipedia articles related to Brazilian higher education system. As far as we know, this is the first Portuguese reading comprehension dataset in this format. We trained a state-of-the-art model on this dataset, which is based on the Bi-Directional Attention Flow model [Seo et al. 2016]. We report on several ablation tests to assess different aspects of both the model and the dataset. For instance, we report learning curves to assess the amount of training data, the use of different levels of pre-trained models, and the use of more than one correct answer for each question.},
   author = {Helio Fonseca Sayama and Anderson Vicoso Araujo and Eraldo Rezende Fernandes},
   doi = {10.1109/BRACIS.2019.00084},
   isbn = {9781728142531},
   journal = {Proceedings - 2019 Brazilian Conference on Intelligent Systems, BRACIS 2019},
   keywords = {Dataset,Machine Reading Comprehension,Natural Language Processing},
   month = {10},
   pages = {443-448},
   publisher = {Institute of Electrical and Electronics Engineers Inc.},
   title = {FaQuAD: Reading comprehension dataset in the domain of brazilian higher education},
   year = {2019},
}
"""

_DESCRIPTION = """\
The FaQuAD is a Portuguese reading comprehension dataset which follows the format of the \
Stanford Question Answering Dataset (SQuAD). As far as we know, FaQuAD is a pioneer Portuguese \
reading comprehension dataset with the SQuAD's challenging format.
"""

_URLS = {
    "train": "https://raw.githubusercontent.com/liafacom/faquad/master/data/train.json",
    "dev": "https://raw.githubusercontent.com/liafacom/faquad/master/data/dev.json",
}


class FaquadConfig(datasets.BuilderConfig):
    """BuilderConfig for FaQuAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for FaQuAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FaquadConfig, self).__init__(**kwargs)


class Faquad(datasets.GeneratorBasedBuilder):
    """FaQuAD: Reading Comprehension Dataset in the Domain of Brazilian Higher Education."""

    BUILDER_CONFIGS = [
        FaquadConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
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
            homepage="https://github.com/liafacom/faquad/",
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
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["dev"]},
            ),
        ]

    def _generate_examples(self, **kwargs):
        """This function returns the examples in the raw (text) form."""
        filepath = kwargs.get("filepath")

        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            for article in squad["data"]:
                title = article.get("title", "")
                for paragraph in article["paragraphs"]:
                    context = paragraph[
                        "context"
                    ]  # do not strip leading blank spaces GH-2585
                    for qa in paragraph["qas"]:
                        answer_starts = [
                            answer["answer_start"] for answer in qa["answers"]
                        ]
                        answers = [answer["text"] for answer in qa["answers"]]
                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield key, {
                            "title": title,
                            "context": context,
                            "question": qa["question"],
                            "id": qa["id"],
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
                        key += 1
