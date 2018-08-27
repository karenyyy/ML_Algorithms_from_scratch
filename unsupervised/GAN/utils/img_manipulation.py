import numpy as np
import matplotlib.pyplot as plt
from utils.visualize import show_image


def apply_gaussian_noise(X, sigma=0.1):
    noise = np.random.normal(loc=0.0, scale=sigma, size=X.shape)
    return X + noise


# test different noise scales
def plot_noisy_imgs(X):
    plt.subplot(1, 4, 1)
    show_image(X[0])
    plt.subplot(1, 4, 2)
    show_image(apply_gaussian_noise(X[0], sigma=0.01))
    plt.subplot(1, 4, 3)
    show_image(apply_gaussian_noise(X[:1], sigma=0.1)[0])
    plt.subplot(1, 4, 4)
    show_image(apply_gaussian_noise(X[:1], sigma=0.5)[0])
    plt.show()

