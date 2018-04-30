from . import abc_model
from . import config

import keras
from keras.models      import Sequential
from keras.layers      import Dense, Dropout, Input
from keras.models      import Model


class Mnist(abc_model.ABCModel):
    @classmethod
    def make_model(cls):
        input_layer = Input(shape=(784,))
        layer2 = Dense(512, activation='relu')(input_layer)
        layer2 = Dropout(0.2)(layer2)
        layer3 = Dense(512, activation='relu')(layer2)
        layer3 = Dropout(0.2)(layer3)
        output = Dense(config.Config.num_classes, activation='softmax')(layer3)
        model = Model(input_layer, output)
        model.summary()

        model.compile(loss=config.Config.loss,
                      optimizer=config.Config.optimizer,
                      metrics=[config.Config.metrics])
        return model

    @classmethod
    def save_model(cls, model):
        print("save"+config.Config.save_model)
        model.save(config.Config.save_model)

    @classmethod
    def load_model(cls):
        print("load"+config.Config.save_model)
        from keras.models import load_model
        return load_model(config.Config.save_model)
