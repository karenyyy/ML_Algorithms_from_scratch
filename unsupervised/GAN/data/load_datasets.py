import pandas as pd
import tarfile
import tqdm
import cv2
import numpy as np
import os
from keras.datasets import fashion_mnist
import tensorflow as tf
from sklearn.model_selection import train_test_split

# download from http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt
ATTRS_NAME = "/home/karen/Downloads/data/lfw_attributes.txt"
# download http://vis-www.cs.umass.edu/lfw/lfw.tgz
IMAGES_FILE = "/home/karen/Downloads/data/lfw.tgz"


def load_lfw(dx=80,
             dy=80,
             h=45,
             w=45):
    def img_preprocess(img):
        # need to decode from bytes to arrays
        decoded_img = cv2.imdecode(np.asarray(bytearray(img), dtype=np.uint8), 1)
        # then transform from BGR to RGB
        processed_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)  # (250, 250, 3)
        cropped_img = processed_img[dy:-dy, dx:-dx]  # (90, 90, 3)
        reshaped_img = cv2.resize(cropped_img, (h, w))  # (45, 45, 3)
        return reshaped_img

    lfw_attributes = pd.read_csv(filepath_or_buffer=ATTRS_NAME, sep='\t', skiprows=1)
    lfw_attributes = pd.DataFrame(lfw_attributes.iloc[:, :-1].values, columns=lfw_attributes.columns[1:])
    person_img_pair = set(map(tuple, lfw_attributes[['person', 'imagenum']].values))

    img_collection = []
    person_img_collection = []

    with tarfile.open(IMAGES_FILE) as file:
        for i in tqdm.tqdm_notebook(file.getmembers()):
            if i.isfile() and i.name.endswith('.jpg'):
                raw_byte_img = file.extractfile(i).read()  # binary file format
                # prepare the images
                preprocessed_img = img_preprocess(raw_byte_img)
                # prepare the text
                img_filename = os.path.split(i.name)[1]
                person_name = img_filename.split('.')[0][:-4].replace('_', ' ').strip()
                img_id = int(img_filename.split('.')[0][-4:])
                # print((person_name, img_id))
                if (person_name, img_id) in person_img_pair:
                    img_collection.append(preprocessed_img)
                    person_img_collection.append({'person': person_name,
                                                  'imagenum': img_id})

    person_img_collection = pd.DataFrame(person_img_collection)
    img_collection = np.stack(img_collection).astype('uint8')
    person_img_collection = person_img_collection.merge(lfw_attributes, on=('person', 'imagenum')).drop(
        ['person', 'imagenum'], axis=1)

    # center images in collection
    centered_imgs = img_collection.astype('float32') / 255.0 - 0.5

    # train test split
    train_imgs, test_imgs = train_test_split(centered_imgs, test_size=0.1, random_state=42)
    return centered_imgs, train_imgs, test_imgs, person_img_collection


def load_fashion_mnist():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    # or keras.utils.to_categorical(y_train, num_classes=..)
    y_train_one_hot = tf.one_hot(y_train, depth=10, axis=1)
    y_test_one_hot = tf.one_hot(y_test, depth=10, axis=1)
    return (X_train, y_train_one_hot), (X_test, y_test_one_hot)


def get_mnist_batches(batch_size=128):
    """

    :param batch_size: default 128
    :return:
    batch_X: Tensor("shuffle_batch:0", shape=(batch_size, h, w), dtype=uint8)
    batch_y: Tensor("shuffle_batch:1", shape=(batch_size, num_classes), dtype=float32)
    """
    (X_train, y_train_one_hot), (X_test, y_test_one_hot) = load_fashion_mnist()
    # print(X_train.shape, X_test.shape, y_train_one_hot.shape, y_test_one_hot.shape)
    training_data = tf.train.slice_input_producer([X_train, y_train_one_hot])
    batch_X, batch_y = tf.train.shuffle_batch(training_data,
                                              batch_size=batch_size,
                                              num_threads=8,
                                              capacity=batch_size * 64,
                                              min_after_dequeue=batch_size * 32,
                                              allow_smaller_final_batch=False)
    return batch_X, batch_y


if __name__ == '__main__':
    X_train, X_test, attr = load_lfw()
    print(X_train.shape, X_test.shape)