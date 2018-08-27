import tensorflow as tf

from data.load_datasets import load_lfw

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.333)
s = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

import keras
from keras.models import Sequential
from keras import layers as L
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tnrange
import os
import cv2

h = 36
w = 36
c = 3

data, _, _, _ = load_lfw(h=h, w=w)


class Gan(object):
    def __init__(self):
        self.height = h
        self.width = w
        self.channels = c
        self.noise_shape = 256
        self.epoches = 50000
        self.batch_size = 100
        self.img_shape = (self.height, self.width, self.channels)
        self.output_dir = 'output_images'
        self.generator = self.Generator()
        self.discriminator = self.Discriminator()

        self.noise = tf.placeholder('float32', [None, self.noise_shape])

        self.real_data = tf.placeholder('float32', [None, ] + list(self.img_shape))

        self.logp_real = self.discriminator(self.real_data)

        self.generated_data = self.generator(self.noise)

        self.logp_gen = self.discriminator(self.generated_data)

        self.d_loss = -tf.reduce_mean(self.logp_real[:, 1] + self.logp_gen[:, 0])

        # regularize
        self.d_loss += tf.reduce_mean(self.discriminator.layers[-1].kernel ** 2)

        # optimize
        self.disc_optimizer = tf.train.GradientDescentOptimizer(1e-3).minimize(self.d_loss,
                                                                               var_list=self.discriminator.trainable_weights)

        g_loss = -tf.reduce_mean(self.logp_gen[:, 1])

        self.gen_optimizer = tf.train.AdamOptimizer(1e-4).minimize(g_loss, var_list=self.generator.trainable_weights)

    def Generator(self):
        generator = Sequential()
        generator.add(L.InputLayer([256], name='noise'))
        generator.add(L.Dense(10 * 8 * 8, activation='elu'))

        generator.add(L.Reshape((8, 8, 10)))
        generator.add(L.Deconv2D(64, kernel_size=(5, 5), activation='elu'))
        generator.add(L.Deconv2D(64, kernel_size=(5, 5), activation='elu'))
        generator.add(L.UpSampling2D(size=(2, 2)))
        generator.add(L.Deconv2D(32, kernel_size=3, activation='elu'))
        generator.add(L.Deconv2D(32, kernel_size=3, activation='elu'))
        generator.add(L.Deconv2D(32, kernel_size=3, activation='elu'))

        generator.add(L.Conv2D(3, kernel_size=3, activation=None))
        generator.summary()
        return generator

    def Discriminator(self):
        discriminator = Sequential()

        discriminator.add(L.InputLayer(self.img_shape))

        discriminator.add(L.Conv2D(8, (3, 3)))
        discriminator.add(L.LeakyReLU(0.1))
        discriminator.add(L.Conv2D(16, (3, 3)))
        discriminator.add(L.LeakyReLU(0.1))
        discriminator.add(L.MaxPool2D())
        discriminator.add(L.Conv2D(32, (3, 3)))
        discriminator.add(L.LeakyReLU(0.1))
        discriminator.add(L.Conv2D(64, (3, 3)))
        discriminator.add(L.LeakyReLU(0.1))
        discriminator.add(L.MaxPool2D())

        discriminator.add(L.Flatten())
        discriminator.add(L.Dense(256, activation='tanh'))
        discriminator.add(L.Dense(2, activation=tf.nn.log_softmax))
        discriminator.summary()
        return discriminator

    def sample_noise_batch(self, batch_size):
        return np.random.normal(size=(batch_size, self.noise_shape)).astype('float32')

    def sample_data_batch(self, batch_size):
        idxs = np.random.choice(np.arange(data.shape[0]), size=batch_size)
        return data[idxs]

    def sample_images(self, nrow, ncol, epoch, sharp=False):
        images = self.generator.predict(self.sample_noise_batch(batch_size=nrow * ncol))
        if np.var(images) != 0:
            images = images.clip(np.min(data), np.max(data))

        if not os.path.exists(path=self.output_dir):
            os.mkdir(path=self.output_dir)

        fig = plt.figure()
        for i in range(nrow * ncol):
            ax = fig.add_subplot(nrow, ncol, i + 1)
            if sharp:
                ax.imshow(images[i].reshape(self.img_shape), cmap="gray", interpolation="none")
            else:
                ax.imshow(images[i].reshape(self.img_shape), cmap="gray")
        fig.savefig(fname=self.output_dir + '/{}.png'.format(epoch))
        plt.show()

    def sample_probas(self, batch_size):
        plt.title('Generated vs real data')
        plt.hist(np.exp(self.discriminator.predict(self.sample_data_batch(batch_size)))[:, 1],
                 label='D(x)', alpha=0.5, range=[0, 1])
        plt.hist(np.exp(self.discriminator.predict(self.generator.predict(self.sample_noise_batch(batch_size))))[:, 1],
                 label='D(G(z))', alpha=0.5, range=[0, 1])
        plt.legend(loc='best')
        plt.show()

    def train(self):
        s.run(tf.global_variables_initializer())
        for epoch in tnrange(self.epoches):
            feed_dict = {
                self.real_data: self.sample_data_batch(batch_size=self.batch_size),
                self.noise: self.sample_noise_batch(batch_size=self.batch_size)
            }

            for i in range(5):
                s.run(self.disc_optimizer, feed_dict)

            s.run(self.gen_optimizer, feed_dict)

            if epoch % self.batch_size == 0:
                print("After " + str(epoch) + " epochs")
                self.sample_images(2, 3, epoch=epoch, sharp=True)
                self.sample_probas(batch_size=1000)
