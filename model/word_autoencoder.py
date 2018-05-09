from . import abc_model
from . import config

import keras
from keras.layers import Dense, Dropout, Input, LSTM
from keras.layers.wrappers import TimeDistributed as TD
from keras.models import Model


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
    def encoder(cls):
        K.set_learning_phase(1)  # set learning phase
        latent_dim = 256
        encoder_inputs = Input(shape=(None, 128))
        encoder_dense_outputs = Dense(128,
                                      activation='sigmoid')(encoder_inputs)
        encoder_lstm_outputs = LSTM(latent_dim, return_sequences=True,
                                    dropout=0.6, recurrent_dropout=0.6)(encoder_dense_outputs)

        decoder_dense_outputs = Dense(
            self.input_dim, activation='sigmoid')(decoder_inputs)
        decoder_lstm = LSTM(
            latent_dim, return_sequences=True, return_state=True)
        decoder_lstm_outputs, _, _ = decoder_lstm(decoder_dense_outputs,
                                                  initial_state=encoder_states)
        decoder_dense_outputs = Dense(self.output_dim,
                                      activation='relu')(decoder_lstm_outputs)
        decoder_outputs = Dense(self.output_dim,
                                activation='linear')(decoder_dense_outputs)

            return Model(encoder_inputs, decoder_outputs)

    @classmethod
    def save_model(cls, model):
        print("save" + config.Config.save_model)
        model.save(config.Config.save_model)

    @classmethod
    def load_model(cls):
        print("load" + config.Config.save_model)
        from keras.models import load_model
        return load_model(config.Config.save_model)
