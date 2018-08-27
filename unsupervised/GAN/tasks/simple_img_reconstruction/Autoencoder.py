from keras.models import Sequential
from keras.models import Model
import keras.layers as L
import keras.backend as K
import tensorflow as tf
import numpy as np
from data.load_datasets import load_lfw
import os
import matplotlib.pyplot as plt
from sklearn.neighbors.unsupervised import NearestNeighbors

from utils.img_manipulation import apply_gaussian_noise
from utils.keras_utils import TqdmProgressCallback
from utils.visualize import show_image

X_train, X_test, _ = load_lfw(dimx=32, dimy=32)
h, w, c = X_train[0].shape


class AutoEncoder(object):
    def __init__(self, h, w, filter_num, task):
        self.height = h
        self.width = w
        self.channel = 3
        self.img_shape = (self.height, self.width, self.channel)
        self.task = task
        self.filter_num = filter_num
        self.encoder, self.decoder, self.autoencoder = self.Combined()

    def reset_tf_session(self):
        K.clear_session()
        tf.reset_default_graph()
        s = K.get_session()
        return s

    def Encoder(self):
        encoder = Sequential()
        encoder.add(L.InputLayer(self.img_shape))

        encoder.add(L.Conv2D(self.filter_num, (3, 3), strides=(1, 1), padding='same', activation='elu'))
        encoder.add(L.MaxPooling2D(pool_size=(2, 2)))
        encoder.add(L.Conv2D(self.filter_num * 2, (3, 3), strides=(1, 1), padding='same', activation='elu'))
        encoder.add(L.MaxPooling2D(pool_size=(2, 2)))
        encoder.add(L.Conv2D(self.filter_num * 4, (3, 3), strides=(1, 1), padding='same', activation='elu'))
        encoder.add(L.MaxPooling2D(pool_size=(2, 2)))
        encoder.add(L.Conv2D(self.filter_num * 8, (3, 3), strides=(1, 1), padding='same', activation='elu'))
        encoder.add(L.MaxPooling2D(pool_size=(2, 2)))
        encoder.add(L.Flatten())
        encoder.add(L.Dense(self.filter_num))
        return encoder

    def Decoder(self):
        decoder = Sequential()
        decoder.add(L.InputLayer((self.filter_num,)))

        decoder.add(L.Dense(np.prod(self.filter_num * 8 * 2 * 2), activation='elu'))
        decoder.add(L.Reshape((2, 2, self.filter_num * 8)))
        decoder.add(
            L.Conv2DTranspose(filters=self.filter_num * 4, kernel_size=(3, 3), strides=2, activation='elu',
                              padding='same'))
        decoder.add(
            L.Conv2DTranspose(filters=self.filter_num * 2, kernel_size=(3, 3), strides=2, activation='elu',
                              padding='same'))
        decoder.add(
            L.Conv2DTranspose(filters=self.filter_num, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
        decoder.add(L.Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=2, activation=None, padding='same'))
        return decoder

    def Combined(self):
        # encoder and decoder frameworks
        encoder = self.Encoder()
        decoder = self.Decoder()

        # prepare the model pipeline
        input_placeholder = L.Input(self.img_shape)
        encoded = encoder(input_placeholder)
        decoded = decoder(encoded)

        autoencoder = Model(inputs=input_placeholder, outputs=decoded)

        return encoder, decoder, autoencoder

    def load_weights(self,
                     encoderfile='encoder.h5',
                     decoderfile='decoder.h5'):
        has_weights = False
        if os.path.isfile(encoderfile) and os.path.isfile(decoderfile):
            has_weights = True
            self.encoder.load_weights(filepath=encoderfile)
            self.decoder.load_weights(filepath=decoderfile)

            inp = L.Input(self.img_shape)
            encoded = self.encoder(inp)
            reconstruction = self.decoder(encoded)
            self.autoencoder = Model(inputs=inp, outputs=reconstruction)

        self.autoencoder.compile(optimizer='adamax', loss='mse')
        return has_weights

    def save_weights(self):
        # save trained weights
        self.encoder.save_weights("encoder.h5")
        self.decoder.save_weights("decoder.h5")

    def train(self):

        if self.task == 'reconstruction':
            has_weights = self.load_weights()
            if not has_weights:
                # fit
                self.autoencoder.fit(x=X_train, y=X_train, epochs=25,
                                     validation_data=[X_test, X_test],
                                     verbose=0)
                self.save_weights()

            # evaluate
            reconstructed_MSE = self.autoencoder.evaluate(x=X_test, y=X_test)
            print('reconstructed_MSE is: ', reconstructed_MSE)
            self.plot_results(X_test)

        elif self.task == 'denoising':
            X_train_noise = apply_gaussian_noise(X_train)
            X_test_noise = apply_gaussian_noise(X_test)

            has_weights = self.load_weights()
            if not has_weights:
                # train the same model with new noise-augmented data
                self.autoencoder.fit(x=X_train_noise, y=X_train, epochs=1,
                                     validation_data=[X_test_noise, X_test],
                                     callbacks=[TqdmProgressCallback()],
                                     verbose=0)
                self.save_weights()

            denoising_mse = self.autoencoder.evaluate(X_test_noise, X_test, verbose=0)
            print("Denoising MSE:", denoising_mse)
            for i in range(5):
                self.plot_results(X_test_noise)

        elif self.task == 'image retrieval':
            has_weights = self.load_weights()
            if not has_weights:
                # fit
                self.autoencoder.fit(x=X_train, y=X_train, epochs=25,
                                     validation_data=[X_test, X_test],
                                     verbose=0)
                self.save_weights()
            encoded = self.encoder.predict(X_train)
            clf = NearestNeighbors(metric="euclidean")
            clf.fit(encoded)
            self.show_similar(clf, self.sample_image(X_test))
        elif self.task == 'image morphing':
            has_weights = self.load_weights()
            if not has_weights:
                # fit
                self.autoencoder.fit(x=X_train, y=X_train, epochs=25,
                                     validation_data=[X_test, X_test],
                                     verbose=0)
                self.save_weights()
            for _ in range(5):
                image1, image2 = X_test[np.random.randint(0, len(X_test), size=2)]

                code1, code2 = self.encoder.predict(np.stack([image1, image2]))

                plt.figure(figsize=[10, 4])
                for i, a in enumerate(np.linspace(0, 1, num=7)):
                    output_code = code1 * (1 - a) + code2 * (a)
                    output_image = self.decoder.predict(output_code[None])[0]

                    plt.subplot(1, 7, i + 1)
                    show_image(output_image)
                    plt.title("a=%.2f" % a)

                plt.show()

    def show_similar(self, clf, X, n_neighbors=5):
        code = self.encoder.predict(X[None])
        (distances,), (idx,) = clf.kneighbors(code, n_neighbors=n_neighbors)
        neighbors = X_train[idx]  # find the most similar images from the training image set
        plt.figure(figsize=[8, 7])
        plt.subplot(1, 4, 1)
        show_image(X)
        plt.title("Original image")

        for i in range(3):
            plt.subplot(1, 4, i + 2)
            show_image(neighbors[i])
            plt.title("Dist=%.3f" % distances[i])
        plt.show()

    def sample_image(self, X):
        # randomly sample an img
        index = np.random.choice(X.shape[0])
        sample_img = X[index]
        return sample_img

    def plot_results(self, X):
        sample_img = self.sample_image(X)

        sample_encoded = self.encoder.predict(x=sample_img[None])[0]
        print(sample_encoded.shape)

        reconstructed_img = self.decoder.predict(x=sample_encoded[None])[0]

        plt.subplot(1, 3, 1)
        plt.imshow(np.clip(sample_img, 0, 1))

        plt.subplot(1, 3, 2)
        plt.imshow(sample_encoded.reshape([sample_encoded.shape[-1] // 2, -1]))

        plt.subplot(1, 3, 3)
        plt.imshow(reconstructed_img)

        plt.show()
