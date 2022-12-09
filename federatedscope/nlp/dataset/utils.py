"""
Utils for language models.
from https://github.com/litian96/FedProx/blob/ \
master/flearn/utils/language_utils.py
"""

import re
import numpy as np
from collections import Counter

# ------------------------
# utils for shakespeare dataset

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[" \
              "]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)


def _one_hot(index, size):
    '''returns one-hot vector with given size and value 1 at given index
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    index = ALL_LETTERS.find(letter)
    return index


def word_to_indices(word):
    '''returns a list of character indices
    Arguments:
        word: string

    :returns:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices


# ------------------------
# utils for sent140 dataset


def split_line(line):
    '''split given line/phrase into list of words
    Arguments:
        line: string representing phrase to be split

    :returns:
        list of strings, with each string representing a word
    '''
    return re.findall(r"[\w']+|[.,!?;]", line)


def bag_of_words(line, vocab):
    '''returns bag of words representation of given phrase using given vocab
    Arguments:
        line: string representing phrase to be parsed
        vocab: dictionary with words as keys and indices as values
    :returns:
        integer list
    '''
    bag = [0] * len(vocab)
    words = split_line(line)
    for w in words:
        if w in vocab:
            bag[vocab[w]] += 1
    return bag


def target_to_binary(label):
    return int(label == 1)


def token_to_ids(texts, vocab):
    to_ret = [[vocab[word] for word in line] for line in texts]
    return np.array(to_ret)


def label_to_index(labels):
    counter = Counter(labels)
    sorted_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    label_list = [x[0] for x in sorted_tuples]
    return [label_list.index(x) for x in labels]
