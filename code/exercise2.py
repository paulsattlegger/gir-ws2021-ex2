"""
Add the code for the 2nd exercise to this file. You can add additional ".py" files to your code repository (e.g. helper
functions, etc.).
"""
import csv
from dataclasses import dataclass
from typing import Generator, Iterable

from gensim.parsing.preprocessing import *
from scipy.stats.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_filters = [strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_short, lower_to_unicode]


@dataclass
class Data:
    ground_truth: float
    text1: str
    text2: str


def read_dataset(file: str) -> Generator[Data, None, None]:
    with open(file) as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        for row in reader:
            yield Data(float(row[0]), row[1], row[2])


def preprocess_dataset(dataset: Iterable[Data]) -> Iterable[Data]:
    for data in dataset:
        data.text1 = preprocess_string(data.text1, filters=_filters)
        data.text2 = preprocess_string(data.text2, filters=_filters)
        yield data


def short_text_embedding_1(data: Data):
    vectorizer = TfidfVectorizer(stop_words='english')
    # Infer the text vector representation for the text pairs in "dataset.tsv".
    X = vectorizer.fit_transform([data.text1, data.text2])
    # Compute the similarity between the text pairs using the cosine similarity.
    return cosine_similarity(X)


def short_text_embedding_2(data: Data):
    pass


def short_text_embedding_3(data: Data):
    pass


def main():
    dataset = list(read_dataset('../dataset.tsv'))
    predicted_scores = list(map(short_text_embedding_1, dataset))
    gt_scores = list(map(lambda d: d.ground_truth, dataset))
    pearson_correlation = pearsonr(gt_scores, predicted_scores)

    dataset = list(preprocess_dataset(dataset))


if __name__ == '__main__':
    main()
