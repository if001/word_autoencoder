# from abc_preprocessing import ABCPreProcessing
from . import abc_preprocessing

import sys
sys.path.append("../")
sys.path.append("../../")
import keras
import random as rand
from itertools import chain
import numpy as np

from cnn_autoencoder import get_feature
from cnn_autoencoder.model.simple_autoencoder import SimpleAutoencoder

num_classes = 10


class PreProcessing(abc_preprocessing.ABCPreProcessing):
    @classmethod
    def __get_word_lists(cls, file_path):
        print("make wordlists")
        with open(file_path) as f:
            lines = f.read().split("\n")
        word_lists = []
        for line in lines:
            word_lists.append(line.split(" "))
        print("wordlist num:", len(word_lists))
        return word_lists[:-1]

    @classmethod
    def __to_uniq(cls, word_lists):
        word_flat_list = list(chain.from_iterable(word_lists))
        word_uniq_lists = list(set(word_flat_list))
        if ' ' in word_uniq_lists:
            word_uniq_lists.remove(' ')
        if '' in word_uniq_lists:
            word_uniq_lists.remove('')
        return word_uniq_lists

    @classmethod
    def make_train_data(cls, data_size, word_len):
        word_list = PreProcessing.__get_word_lists(
            "../aozora_data/files/files_all_rnp.txt")
        uniq_word_lists = PreProcessing.__to_uniq(word_list)
        print("uniq word len:", len(uniq_word_lists))

        autoencoder = SimpleAutoencoder.load_model("autoencoder.hdf5")
        encoder = SimpleAutoencoder.make_encoder_model(autoencoder)

        train_x = []
        train_y = []
        for i in range(data_size):
            rand_num = rand.randint(0, len(uniq_word_lists) - 1)
            select_word = uniq_word_lists[rand_num]
            while(len(select_word) != word_len):
                rand_num = rand.randint(0, len(uniq_word_lists) - 1)
                select_word = uniq_word_lists[rand_num]
            word_feature = []
            for char in select_word:
                feature = get_feature.char2feature(char, encoder)
                feature = feature.reshape(4 * 4 * 8)
                word_feature.append(feature)

            if len(word_feature) == 1:
                train_x.append(word_feature)
                train_y.append(word_feature)
            else:
                train_x.append(word_feature[:-1])
                train_y.append(word_feature[1:])

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        print("train_x shape:", train_x.shape)
        print("train_y shape:", train_y.shape)
        return train_x, train_y

    @classmethod
    def make_test_data(cls):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = x_test.reshape(10000, 784)
        x_test = x_test.astype('float32')
        x_test /= 255
        print(x_test.shape[0], 'test samples')
        y_test = keras.utils.to_categorical(y_test, num_classes)
        return x_test, y_test


def main():
    pass


if __name__ == '__main__':
    main()
