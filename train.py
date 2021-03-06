from model.word_autoencoder import WordAutoencoder
from preprocessing.preprocessing import PreProcessing
from model_exec.learning import Learning
from model_exec.predict import Predict
import numpy as np

data_size = 3


def main():
    train_x, train_y = PreProcessing().make_train_data(data_size, word_len=2)
    word_autoencoder_model = WordAutoencoder().make_model()
    cbs = WordAutoencoder().set_callbacks("model.hdf5")
    hist = Learning.run(word_autoencoder_model, train_x, train_y, cbs)
    # WordAutoencoder.save_model(word_autoencoder_model)

    print("test:", train_x[0])
    test_x = np.array([train_x[0]])
    score = Predict.run(word_autoencoder_model, test_x)
    print("score:", score)


if __name__ == '__main__':
    main()
