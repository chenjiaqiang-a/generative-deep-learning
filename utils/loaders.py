import os
import pickle
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

__all__ = ["load_model", "load_mnist", "ImageLabelLoader"]


class ImageLabelLoader:
    def __init__(self, image_folder, target_size):
        self.image_folder = image_folder
        self.target_size = target_size
    
    def build(self, att, batch_size, label=None):
        data_gen = ImageDataGenerator(rescale=1.0/255.0)
        if label:
            data_flow = data_gen.flow_from_dataframe(
                att,
                self.image_folder,
                x_col="image_id",
                y_col=label,
                target_size=self.target_size,
                class_mode="raw",
                batch_size=batch_size,
                shuffle=True
            )
        else:
            data_flow = data_gen.flow_from_dataframe(
                att,
                self.image_folder,
                x_col="image_id",
                target_size=self.target_size,
                class_mode="input",
                batch_size=batch_size,
                shuffle=True
            )
        return data_flow


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