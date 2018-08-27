from keras.models import Sequential, Model
import keras.layers as L
from keras.datasets import cifar100
import keras

(X_train, y_train), (_, _) = cifar100.load_data()

h, w, c = X_train[0].shape

mask_h = 8
mask_w = 8


class ContextEncoder(object):
    def __init__(self):
        self.height = h
        self.width = w
        self.channels = c
        self.mask_height = mask_h
        self.mask_width = mask_w
        self.img_shape = (self.height, self.width, self.channels)
        self.missing_shape = (self.mask_height, self.mask_width, self.channels)

        self.optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        keras.optimizers.Adam()

        # build discriminator
        self.discriminator = self.Discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])

        # build the generator
        self.generator = self.Generator()

        # The generator takes noise as input and generates the missing part of the image
        masked_img = L.Input(shape=self.img_shape)
        gen_missing = self.generator(masked_img)

        self.discriminator.trainable = False

        # input corrupted images into the discriminator
        valid = self.discriminator(gen_missing)


        self.combined = Model(masked_img, [gen_missing, valid])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
                              loss_weights=[0.999, 0.001],
                              optimizer=self.optimizer)

    def Generator(self):
        generator = Sequential()

        # Encoder
        generator.add(L.Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        generator.add(L.LeakyReLU(alpha=0.2))
        generator.add(L.BatchNormalization(momentum=0.8))
        generator.add(L.Conv2D(64, kernel_size=3, strides=2, padding="same"))
        generator.add(L.LeakyReLU(alpha=0.2))
        generator.add(L.BatchNormalization(momentum=0.8))
        generator.add(L.Conv2D(128, kernel_size=3, strides=2, padding="same"))
        generator.add(L.LeakyReLU(alpha=0.2))
        generator.add(L.BatchNormalization(momentum=0.8))

        generator.add(L.Conv2D(512, kernel_size=1, strides=2, padding="same"))
        generator.add(L.LeakyReLU(alpha=0.2))
        generator.add(L.Dropout(0.5))

        # Decoder
        generator.add(L.UpSampling2D())
        generator.add(L.Conv2D(128, kernel_size=3, padding="same"))
        generator.add(L.Activation('relu'))
        generator.add(L.BatchNormalization(momentum=0.8))
        generator.add(L.UpSampling2D())
        generator.add(L.Conv2D(64, kernel_size=3, padding="same"))
        generator.add(L.Activation('relu'))
        generator.add(L.BatchNormalization(momentum=0.8))
        generator.add(L.Conv2D(self.channels, kernel_size=3, padding="same"))
        generator.add(L.Activation('tanh'))

        generator.summary()

        masked_img = L.Input(shape=self.img_shape)
        gen_missing = generator(masked_img)

        return Model(masked_img, gen_missing)

    def Discriminator(self):
        discriminator = Sequential()

        discriminator.add(L.Conv2D(64, kernel_size=3, strides=2, input_shape=self.missing_shape, padding="same"))
        discriminator.add(L.LeakyReLU(alpha=0.2))
        discriminator.add(L.BatchNormalization(momentum=0.8))
        discriminator.add(L.Conv2D(128, kernel_size=3, strides=2, padding="same"))
        discriminator.add(L.LeakyReLU(alpha=0.2))
        discriminator.add(L.BatchNormalization(momentum=0.8))
        discriminator.add(L.Conv2D(256, kernel_size=3, padding="same"))
        discriminator.add(L.LeakyReLU(alpha=0.2))
        discriminator.add(L.BatchNormalization(momentum=0.8))
        discriminator.add(L.Flatten())
        discriminator.add(L.Dense(1, activation='sigmoid'))
        discriminator.summary()

        img = L.Input(shape=self.missing_shape)
        validity = discriminator(img)

        return Model(img, validity)

    def train(self):
        return NotImplementedError

    def mask(self):
        return NotImplementedError

    def plot_results(self):
        return NotImplementedError