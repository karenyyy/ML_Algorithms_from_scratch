from data.load_datasets import load_lfw
from tasks.simple_img_reconstruction.Autoencoder import AutoEncoder

X_train, X_test, _ = load_lfw(dimx=32, dimy=32)
h, w, c = X_train[0].shape

# task = ['reconstruction', 'denoising', 'image retrieval', 'image morphing']
#
# for i in task:
#     auto = AutoEncoder(h, w, filter_num=32, task=i)
#     auto.train()
#
auto = AutoEncoder(h, w, filter_num=32, task='image retrieval')
auto.train()
