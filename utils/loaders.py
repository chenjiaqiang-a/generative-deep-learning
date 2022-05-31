import os
import pickle
import numpy as np
from tensorflow.keras.datasets import mnist, cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator

__all__ = ["load_model", "load_mnist", "ImageLabelLoader", "load_safari",
           "load_cifar", "load_celeb"]


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


def load_safari(folder):
    path = os.path.join("./data", folder)
    txt_name_list = []
    for (ditpath, dirname, filenames) in os.walk(path):
        for f in filenames:
            if f != '.DS_Store':
                txt_name_list.append(f)
                break
    
    slice_train = int(80000/len(txt_name_list))
    i = 0
    seed = np.random.randint(1, 10e6)

    for txt_name in txt_name_list:
        txt_path = os.path.join(path, txt_name)
        x = np.load(txt_path)
        x = (x.astype("float32") - 127.5) / 127.5
        x = x.reshape(x.shape[0], 28, 28, 1)

        y = [i] * len(x)
        np.random.seed(seed)
        np.random.shuffle(x)
        np.random.seed(seed)
        np.random.shuffle(y)
        x = x[:slice_train]
        y = y[:slice_train]
        if i != 0:
            xtotal = np.concatenate((x, xtotal), axis=0)
            ytotal = np.concatenate((y, ytotal), axis=0)
        else:
            xtotal = x
            ytotal = y
        i += 1
    
    return xtotal, ytotal


def load_cifar(label, num):
    if num == 10:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
    
    train_mask = [y[0] == label for y in y_train]
    test_mask = [y[0] == label for y in y_test]

    x_data = np.concatenate([x_train[train_mask], x_test[test_mask]])
    y_data = np.concatenate([y_train[train_mask], y_test[test_mask]])

    x_data = (x_data.astype("float32") - 127.5) / 127.5

    return (x_data, y_data)


def load_celeb(data_name, image_size, batch_size):
    data_folder = os.path.join("./data", data_name)

    data_gen = ImageDataGenerator(preprocessing_function=lambda x: (x.astype('float32') - 127.5) / 127.5)

    x_train = data_gen.flow_from_directory(
        data_folder,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        class_mode="input",
        subset='training'
    )
    return x_train
