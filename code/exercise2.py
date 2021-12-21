"""
Add the code for the 2nd exercise to this file. You can add additional ".py" files to your code repository (e.g. helper functions, etc.).
"""
from csv import DictReader
from typing import Generator, Iterable

from gensim.parsing.preprocessing import *

_fieldnames = ['ground_truth', 'text1', 'text2']
_filters = [strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_short, lower_to_unicode]


def read_dataset(file: str) -> Generator[dict, None, None]:
    with open(file) as tsv:
        reader = DictReader(tsv, fieldnames=_fieldnames, delimiter='\t')
        for row in reader:
            yield row


def preprocess_dataset(dataset: Iterable[dict]) -> Iterable[dict]:
    for row in dataset:
        row[_fieldnames[1]] = preprocess_string(row[_fieldnames[1]], filters=_filters)
        row[_fieldnames[2]] = preprocess_string(row[_fieldnames[2]], filters=_filters)
        yield row


def main():
    dataset = read_dataset('../dataset.tsv')
    preprocessed_dataset = preprocess_dataset(dataset)
    print(list(preprocessed_dataset))


if __name__ == '__main__':
    main()
