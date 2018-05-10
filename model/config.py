from keras.optimizers import RMSprop
import os


class Config():
    num_classes = 10
    run_dir_path = os.path.dirname(os.path.abspath(__file__))
    save_model = run_dir_path + "/weight/model.hdf5"
    loss = 'mean_squared_error'
    optimizer = 'adam'
    metrics = 'accuracy'
