"""
Add the code for the 2nd exercise to this file. You can add additional ".py" files to your code repository (e.g. helper
functions, etc.).
"""
from __future__ import annotations
import csv
import gzip
import json
import time
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
import numpy as np
from typing import Generator, Iterable, Callable, List, Dict, Any
import gensim
from gensim.models import KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from gensim.parsing.preprocessing import *
from scipy.stats.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

@dataclass
class Data:
    ground_truth: float
    text1: str | list[str]
    text2: str | list[str]

    @cached_property
    def vocabulary(self):
        return list(set(chain(self.text1, self.text2)))


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


def filter_data(data: Data) -> Data:
    return Data(data.ground_truth,
                [word for word in data.text1 if word in language_model.key_to_index],
                [word for word in data.text2 if word in language_model.key_to_index])


def _cosine_similarity(x: Any, y: Any = None, dense_output: Any = True) -> Any:
    # Note: [0, 0] extracts value from ndarray; [] around tfidf is needed, because ndarray is expected
    return cosine_similarity([x], [y], dense_output)[0, 0]


def short_text_embedding_1(data: Data) -> float:
    vectorizer = TfidfVectorizer()
    # Infer the text vector representation for the text pairs in "dataset.tsv".
    # Note: The input is expected to be a sequence of items that can be of type string or byte.
    # Thus, we use ' '.join() here.
    tfidf =  vectorizer.fit_transform([
        ' '.join(data.text1),
        ' '.join(data.text2)
    ]).toarray()
    # Compute the similarity between the text pairs using the cosine similarity.
    return _cosine_similarity(tfidf[0], tfidf[1])


def short_text_embedding_2(data: Data) -> float:
    data = filter_data(data)
    vector_representation = np.zeros((2, len(data.vocabulary)))
    # For each word appearing in a text, compute a word embedding.
    for word in data.text1:
        word_embedding = language_model.get_vector(word, norm=True)
        # The word embeddings are aggregated via mean averaging to infer a vector representation for the text.
        vector_representation[0, data.vocabulary.index(word)] = np.mean(word_embedding)
    # For each word appearing in a text, compute a word embedding.
    for word in data.text2:
        word_embedding = language_model.get_vector(word, norm=True)
        # The word embeddings are aggregated via mean averaging to infer a vector representation for the text.
        vector_representation[1, data.vocabulary.index(word)] = np.mean(word_embedding)

    return _cosine_similarity(vector_representation[0], vector_representation[1])


def short_text_embedding_3(data: Data) -> float:
    data = filter_data(data)
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([
        ' '.join(data.text1),
        ' '.join(data.text2)
    ]).toarray()

    vector_representation = np.zeros_like(tfidf)

    for word in data.text1:
        word_embedding = language_model.get_vector(word, norm=True)
        index = vectorizer.vocabulary_.get(word)
        # vector_representation[0, index] = np.mean(word_embedding)
        vector_representation[0, index] = np.mean(word_embedding) * tfidf[0][index]
    for word in data.text2:
        word_embedding = language_model.get_vector(word, norm=True)
        index = vectorizer.vocabulary_.get(word)
        # vector_representation[1, index] = np.mean(word_embedding)
        vector_representation[1, index] = np.mean(word_embedding) * tfidf[1][index]

    # result1 = np.average(vector_representation[0], weights=tfidf[0])
    # result2 = np.average(vector_representation[1], weights=tfidf[1])
    return cosine_similarity([vector_representation[0]], [vector_representation[1]])[0, 0]
    # return cosine_similarity([result1], [result2])


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

    print(f'\n{" Cosine similarity between the word vectors ".center(153, "#")}\n')
    print('Cosine similarity for pair1: ("cat", "dog")        =', _cosine_similarity(word_vector_cat, word_vector_dog))
    print('Cosine similarity for pair2: ("cat", "Vienna")     =',
          _cosine_similarity(word_vector_cat, word_vector_vienna))
    print('Cosine similarity for pair3: ("Vienna", "Austria") =',
          _cosine_similarity(word_vector_vienna, word_vector_austria))
    print('Cosine similarity for pair4: ("Austria", "dog")    =',
          _cosine_similarity(word_vector_austria, word_vector_dog))

    print(f'\n{" Top-3 most similar words ".center(153, "#")}\n')
    print('Top-3 most similar words for word1: "Vienna"  =',
          language_model.most_similar(positive=['Vienna'], topn=3))
    print('Top-3 most similar words for word2: "Austria" =',
          language_model.most_similar(positive=['Austria'], topn=3))
    print('Top-3 most similar words for word3: "cat":    =', language_model.most_similar(positive=['cat'], topn=3))


def part2():
    print(f'\n{" Short-Text Similarity ".center(153, "#")}\n')
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


def read_json_from_zip() -> Generator[List[Dict], None, None]:
    path = "../dataset/german-tweet-sample-2019-04/"
    for child in Path(path).iterdir():
        if child.is_file():
            with gzip.open(child, 'r') as zipfile:
                yield json.load(zipfile)


def get_german_data_sample() -> Generator[str, None, None]:
    data = read_json_from_zip()
    for sample in data:
        for tweet in sample:
            yield tweet.get("text")


class PrintLoss(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.total_start = time.process_time()
        self.epoch_start = time.process_time()

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        elapsed_t = time.process_time() - self.total_start
        elapsed_e = time.process_time() - self.epoch_start
        print(f'Loss after epoch {self.epoch}: {loss} \n Total time: {elapsed_t} \n Epoch time: {elapsed_e}\n')
        self.epoch_start = time.process_time()
        self.epoch += 1


def train_german_model(model_path):
    filters = [strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, strip_short]
    texts = get_german_data_sample()
    texts_tokenized = [preprocess_string(text.lower(), filters) for text in texts if text is not None]
    print("start training...")
    german_model = gensim.models.Word2Vec(
        sentences=texts_tokenized,
        vector_size=100,
        window=5,
        min_count=5,
        epochs=10,
        workers=4,
        compute_loss=True,
        callbacks=[PrintLoss()])

    print(f"saving model to {model_path}")
    german_model.save(model_path)
    return german_model


def load_german_modal(model_path):
    return gensim.models.Word2Vec.load(model_path)


def part3():
    model_path = "../models/german.model"
    words = ["obst", "universität", "tisch"]
    train = False

    if train:
        train_german_model(model_path)
    german_model = load_german_modal(model_path)

    for word in words:
        similar = german_model.wv.most_similar(positive=[word], topn=3)
        print(f'Top 3 similar words to "{word}" = {similar}')


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
    part3()
