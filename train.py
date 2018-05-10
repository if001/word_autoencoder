from model.word_autoencoder import WordAutoencoder
from preprocessing.preprocessing import PreProcessing
from model_exec.learning import Learning
from model_exec.predict import Predict
import numpy as np


def main():
    train_x, train_y = PreProcessing().make_train_data(data_size=3, word_len=2)
    word_autoencoder_model = WordAutoencoder().make_model()
    hist = Learning.run(word_autoencoder_model, train_x, train_y)
    # WordAutoencoder.save_model(word_autoencoder_model)

    print(train_x)
    test_x = np.array([train_x[0]])
    score = Predict.run(word_autoencoder_model, test_x)
    print(score)


if __name__ == '__main__':
    main()
