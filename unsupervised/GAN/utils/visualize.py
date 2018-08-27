import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2


def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))


def merge_imgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        col = idx % size[1]
        row = idx // size[1]
        imgs[row * h:row * h + h, col * w:col * w + w, :] = image
    return imgs


def save_img(imgs, size, path):
    return cv2.imwrite(path, merge_imgs(imgs, size))
