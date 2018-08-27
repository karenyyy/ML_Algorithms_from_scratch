from keras.datasets import cifar100


(X_train, y_train), (_, _) = cifar100.load_data()

print(X_train.shape)

