"""
Add the code for the 2nd exercise to this file. You can add additional ".py" files to your code repository (e.g. helper functions, etc.).
"""
import gensim.models
from sklearn.metrics.pairwise import cosine_similarity


def part1():
    model = gensim.models.KeyedVectors.load_word2vec_format('../wiki-news-300d-1M-subword.vec')
    # print(model.get_vector("word", norm=True))  # normalizing usually improves performance
    # print(model.get_vector("word", norm=True).__sizeof__())  # normalizing usually improves performance

    word_vector_cat = model.get_vector("cat", norm=True)
    word_vector_dog = model.get_vector("dog", norm=True)
    word_vector_vienna = model.get_vector("Vienna", norm=True)
    word_vector_austria = model.get_vector("Austria", norm=True)

    print('Cosine similarity for pair1: ("cat", "dog") = {}'.format(
        cosine_similarity(word_vector_cat.reshape(1, -1), word_vector_dog.reshape(1, -1))))
    print('Cosine similarity for pair2: ("cat", "Vienna") = {}'.format(
        cosine_similarity(word_vector_cat.reshape(1, -1), word_vector_vienna.reshape(1, -1))))
    print('Cosine similarity for pair3: ("Vienna", "Austria") = {}'.format(
        cosine_similarity(word_vector_vienna.reshape(1, -1), word_vector_austria.reshape(1, -1))))
    print('Cosine similarity for pair4: ("Austria", "dog") = {}'.format(
        cosine_similarity(word_vector_austria.reshape(1, -1), word_vector_dog.reshape(1, -1))))


def main():
    part1()


if __name__ == "__main__":
    main()
