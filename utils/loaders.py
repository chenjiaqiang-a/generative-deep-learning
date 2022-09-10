import os
import pickle
import skimage.transform as trans
from skimage.io import imread
import numpy as np
from glob import glob
from tensorflow.keras.datasets import mnist, cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator

__all__ = ["load_model", "load_mnist", "ImageLabelLoader", "load_safari",
           "load_cifar", "load_celeb", "DataLoader"]


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


class DataLoader:
    def __init__(self, dataset_name, img_res=(256, 256)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = f'train{domain}' if not is_testing else f'test{domain}'
        path = glob(f'./data/{self.dataset_name}/{data_type}/*')

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = trans.resize(img, self.img_res)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = trans.resize(img, self.img_res)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.0

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = 'train' if not is_testing else 'val'
        path_A = glob(f'./data/{self.dataset_name}/{data_type}A/*')
        path_B = glob(f'./data/{self.dataset_name}/{data_type}B/*')

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                img_A = trans.resize(img_A, self.img_res)
                img_B = trans.resize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = trans.resize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return imread(path, pilmode='RGB').astype(np.float32)


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

    data_gen = ImageDataGenerator(preprocessing_function=lambda x: (
        x.astype('float32') - 127.5) / 127.5)

    x_train = data_gen.flow_from_directory(
        data_folder,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        class_mode="input",
        subset='training'
    )
    return x_train
