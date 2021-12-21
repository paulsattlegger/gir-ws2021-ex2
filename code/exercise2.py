"""
Add the code for the 2nd exercise to this file. You can add additional ".py" files to your code repository (e.g. helper
functions, etc.).
"""
import csv
import gensim.models
from dataclasses import dataclass
from typing import Generator, Iterable, Callable

from gensim.parsing.preprocessing import *
from scipy.stats.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def preprocess_dataset(dataset: Iterable[Data], remove_stopwords_: bool = False) -> Iterable[Data]:
    filters = [strip_punctuation, strip_multiple_whitespaces, strip_short, lower_to_unicode]
    if remove_stopwords_:
        filters.append(remove_stopwords)
    for data in dataset:
        yield Data(data.ground_truth,
                   preprocess_string(data.text1, filters=filters),
                   preprocess_string(data.text2, filters=filters))


def short_text_embedding_1(data: Data) -> float:
    vectorizer = TfidfVectorizer()
    # Infer the text vector representation for the text pairs in "dataset.tsv".
    # The input is expected to be a sequence of items that can be of type string or byte. Thus, we use ' '.join() here.
    tfidf = vectorizer.fit_transform([
        ' '.join(data.text1),
        ' '.join(data.text2)
    ]).toarray()
    # Compute the similarity between the text pairs using the cosine similarity.
    return cosine_similarity([tfidf[0]], [tfidf[1]])[0, 0]


def short_text_embedding_2(data: Data) -> float:
    pass


def short_text_embedding_3(data: Data) -> float:
    pass


def evaluate(func: Callable[[Data], float], dataset: Iterable[Data]):
    predicted_scores = list(map(func, dataset))
    gt_scores = list(map(lambda d: d.ground_truth, dataset))
    return pearsonr(gt_scores, predicted_scores)[0]


def part1():
    model = gensim.models.KeyedVectors.load_word2vec_format('../wiki-news-300d-1M-subword.vec')
    # print(model.get_vector("word", norm=True))  # normalizing usually improves performance
    # print(model.get_vector("word", norm=True).__sizeof__())  # normalizing usually improves performance

    word_vector_cat = model.get_vector("cat", norm=True)
    word_vector_dog = model.get_vector("dog", norm=True)
    word_vector_vienna = model.get_vector("Vienna", norm=True)
    word_vector_austria = model.get_vector("Austria", norm=True)

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
        model.most_similar(positive=['Vienna'], topn=3)
    ))
    print('Top-3 most similar words for word2: "Austria": {}'.format(
        model.most_similar(positive=['Austria'], topn=3)
    ))
    print('Top-3 most similar words for word3: "cat": {}'.format(
        model.most_similar(positive=['cat'], topn=3)
    ))


def main():
    part1()
    dataset = list(read_dataset('../dataset.tsv'))
    w_stopword_removal = list(preprocess_dataset(dataset, remove_stopwords_=True))
    wo_stopword_removal = list(preprocess_dataset(dataset))

    print('| Method                           | Preprocessing           | Pearson Correlation |')
    print('|----------------------------------|-------------------------|---------------------|')
    # Lower-casing + Stopword
    print('| Vektor Space Model               | Lower-casing + Stopword |',
          evaluate(short_text_embedding_1, w_stopword_removal), '-|')
    print('| Average Word Embedding           | Lower-casing + Stopword |',
          evaluate(short_text_embedding_2, w_stopword_removal), '-|')
    print('| IDF Weighted Agg. Word Embedding | Lower-casing + Stopword |',
          evaluate(short_text_embedding_3, w_stopword_removal), '-|')

    # Lower-casing
    print('| Vektor Space Model               | Lower-casing            |',
          evaluate(short_text_embedding_1, wo_stopword_removal), '-|')
    print('| Average Word Embedding           | Lower-casing            |',
          evaluate(short_text_embedding_2, wo_stopword_removal), '-|')
    print('| IDF Weighted Agg. Word Embedding | Lower-casing            |',
          evaluate(short_text_embedding_3, wo_stopword_removal), '-|')


if __name__ == '__main__':
    main()
