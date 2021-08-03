import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Planck, speed_of_light, elementary_charge


def l_e(l):
    h = Planck
    c = speed_of_light
    q = elementary_charge

    return (h * c / (l * 1e-9)) / q

def gaussian(x, center, sigma):
    return np.exp(-np.power((x - center) / sigma, 2))

def pulse(xs, start, end):
    def F(x):
        if (x > start) and (x < end):
            return 1.0

        return 0.0

    return np.vectorize(F)(xs)


def normalize(x, y):
    N = np.trapz(y, x, x[1] - x[0])
    return y / N

def plot_2d(x, y, z):
    X,Y = np.meshgrid(x, y)
    plt.contourf(X, Y, z, levels = 100)