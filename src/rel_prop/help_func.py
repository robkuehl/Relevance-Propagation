import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import colors
from src.rel_prop.rel_prop_grad import *

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train/255.
x_test = x_test/255.


class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def pred(model, idx, dataset: str, eps: float, gamma: float):
    if dataset is 'train':
        data_x = x_train
        data_y = y_train
    else:
        data_x = x_test
        data_y = y_test

    prediction = model.predict(np.array([data_x[idx]], dtype=np.dtype(float)))

    print('Correct Label: {}\n'.format(classes[data_y[idx][0]]))
    for i in range(10):
        print('{}:\t\t {:.2f}'.format(classes[i], prediction[0][i]))
    print('\nNetwork Decision: {}'.format(classes[np.argmax(prediction)]))

    timestamp = time.strftime('%d-%m_%Hh%m')
    persist_string = f'{dataset}_{idx}_{timestamp}'
    config_string = f'filename: {dataset}_{idx}\neps: {eps}   gamma: {gamma}'

    img = np.array([data_x[idx].astype(float)])
    mask = np.zeros(10, dtype=np.dtype(float))
    mask[data_y[idx]] = 1.
    mask = tf.constant(mask, dtype=tf.float32)
    relevance = rel_prop(model, img, mask, eps=eps, gamma=gamma)
    relevance = relevance.sum(axis=3)
    plt.suptitle(config_string)
    plt.subplot(1, 2, 1)
    plt.imshow(data_x[idx])
    plt.subplot(1, 2, 2)
    plt.imshow(relevance[0], cmap='seismic',
               norm=MidpointNormalize(midpoint=0, vmin=relevance[0].min(), vmax=relevance[0].max()))

    plt.savefig('figures/' + persist_string)
    plt.show()


classes = {0 : 'airplane',
1 : 'automobile',
2 : 'bird',
3 : 'cat',
4 : 'deer',
5 : 'dog',
6 : 'frog',
7 : 'horse',
8 : 'ship',
9 : 'truck'}

