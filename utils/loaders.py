import os
import pickle
import numpy as np
from tensorflow.keras.datasets import mnist

__all__ = ["load_model", "load_mnist"]


def load_model(model_class, folder):
    with open(os.path.join(folder, "params.pkl"), "rb") as f:
        params = pickle.load(f)
    
    model = model_class(*params)
    model.load_weights(os.path.join(folder, "weights/weights.h5"))

    return model


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.astype("float32") / 255.0
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test.reshape(x_test.shape + (1,))
    
    return (x_train, y_train), (x_test, y_test)