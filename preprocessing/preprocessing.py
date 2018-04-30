# from abc_preprocessing import ABCPreProcessing
from . import abc_preprocessing

import sys
sys.path.append("../")
# from lib.data_shaping import DataShaping
import keras
from keras.datasets    import mnist

num_classes = 10

class PreProcessing(abc_preprocessing.ABCPreProcessing):
    @classmethod
    def make_train_data(cls):
        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_train = x_train.astype('float32')
        x_train /= 255
        print(x_train.shape[0], 'train samples')
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        return x_train, y_train

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
