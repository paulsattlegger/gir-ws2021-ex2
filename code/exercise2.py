"""
Add the code for the 2nd exercise to this file. You can add additional ".py" files to your code repository (e.g. helper
functions, etc.).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Callable

import numpy as np
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import *
from scipy.stats.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Data:
    ground_truth: float
    text1: str | list[str]
    text2: str | list[str]


def read_dataset(file: str) -> Generator[Data, None, None]:
    with open(file) as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        for row in reader:
            yield Data(float(row[0]), row[1], row[2])


def preprocess_dataset(dataset: Iterable[Data], remove_stopwords_: bool = False) -> Iterable[Data]:
    filters = [lower_to_unicode, strip_punctuation, remove_stopwords] if remove_stopwords_ else [lower_to_unicode,
                                                                                                 strip_punctuation]
    for data in dataset:
        yield Data(data.ground_truth,
                   preprocess_string(data.text1, filters=filters),
                   preprocess_string(data.text2, filters=filters))


def short_text_embedding_1(data: Data) -> float:
    vectorizer = TfidfVectorizer()
    # Infer the text vector representation for the text pairs in "dataset.tsv".
    # Note: The input is expected to be a sequence of items that can be of type string or byte.
    # Thus, we use ' '.join() here.
    tfidf = vectorizer.fit_transform([
        ' '.join(data.text1),
        ' '.join(data.text2)
    ]).toarray()
    # Compute the similarity between the text pairs using the cosine similarity.
    # Note: [0, 0] extracts value from ndarray; [] around tfidf is needed, because ndarray is expected
    return cosine_similarity([tfidf[0]], [tfidf[1]])[0, 0]


def compute_word_embedding(text: list[str]) -> Generator[np.array, None, None]:
    # For each word appearing in a text, compute a word embedding.
    for word in text:
        try:
            yield language_model.get_vector(word, norm=True)
        except KeyError:
            pass


def short_text_embedding_2(data: Data) -> float:
    vector_representation = np.zeros((2, max(len(data.text1), len(data.text2))))
    # The word embeddings are aggregated via mean averaging to infer a vector representation for the text.
    for i, vector in enumerate(compute_word_embedding(data.text1)):
        vector_representation[0, i] = np.mean(vector)
    for i, vector in enumerate(compute_word_embedding(data.text2)):
        vector_representation[1, i] = np.mean(vector)

    return cosine_similarity([vector_representation[0]], [vector_representation[1]])[0, 0]


def short_text_embedding_3(data: Data) -> float:
    pass


def evaluate(func: Callable[[Data], float], dataset: Iterable[Data]):
    predicted_scores = list(map(func, dataset))
    gt_scores = list(map(lambda d: d.ground_truth, dataset))
    return pearsonr(gt_scores, predicted_scores)[0]


def part1():
    # Compute the word vectors for each given word
    word_vector_cat = language_model.get_vector("cat", norm=True)
    word_vector_dog = language_model.get_vector("dog", norm=True)
    word_vector_vienna = language_model.get_vector("Vienna", norm=True)
    word_vector_austria = language_model.get_vector("Austria", norm=True)

    print('########## Cosine similarity between the word vectors ##########\n')
    print('Cosine similarity for pair1: ("cat", "dog") = {}'.format(
        cosine_similarity([word_vector_cat], [word_vector_dog])[0][0]))
    print('Cosine similarity for pair2: ("cat", "Vienna") = {}'.format(
        cosine_similarity([word_vector_cat], [word_vector_vienna])[0][0]))
    print('Cosine similarity for pair3: ("Vienna", "Austria") = {}'.format(
        cosine_similarity([word_vector_vienna], [word_vector_austria])[0][0]))
    print('Cosine similarity for pair4: ("Austria", "dog") = {}'.format(
        cosine_similarity([word_vector_austria], [word_vector_dog])[0][0]))

    print('\n########## Top-3 most similar words ##########\n')
    print('Top-3 most similar words for word1: "Vienna": {}'.format(
        language_model.most_similar(positive=['Vienna'], topn=3)
    ))
    print('Top-3 most similar words for word2: "Austria": {}'.format(
        language_model.most_similar(positive=['Austria'], topn=3)
    ))
    print('Top-3 most similar words for word3: "cat": {}'.format(
        language_model.most_similar(positive=['cat'], topn=3)
    ))


def part2():
    dataset = list(read_dataset('../dataset.tsv'))
    w_stopword_removal = list(preprocess_dataset(dataset, remove_stopwords_=True))
    wo_stopword_removal = list(preprocess_dataset(dataset))

    datasets = [
        ['Lower-casing + Stopword', w_stopword_removal],
        ['Lower-casing', wo_stopword_removal]
    ]
    methods = [
        ['Vector Space Model', short_text_embedding_1],
        ['Average Word Embedding', short_text_embedding_2],
        ['IDF Weighted Agg. Word Embedding', short_text_embedding_3]
    ]

    print(f'| {"Method":32} | {"Preprocessing":23} | {"Pearson Correlation":19} |')
    print(f'| {"-" * 32} | {"-" * 23} | {"-" * 19} |')
    for d_text, d in datasets:
        for m_text, m in methods:
            print(f'| {m_text:32} | {d_text:23} | {evaluate(m, d):19} |')


if __name__ == '__main__':
    kv_file = Path('../wiki-news-300d-1M-subword.kv')
    vec_file = Path('../wiki-news-300d-1M-subword.vec')

    if kv_file.exists():
        language_model = KeyedVectors.load(str(kv_file))
    else:
        language_model = KeyedVectors.load_word2vec_format(str(vec_file))
        language_model.save(str(kv_file))

    part1()
    part2()
