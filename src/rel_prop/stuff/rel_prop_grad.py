import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
import sys


def rho(layer, c: int):
    try:
        weights = layer.get_weights()
        layer.set_weights([tf.add(weights[0],tf.clip_by_value(np.multiply(layer.get_weights(), c)[0],
                                            clip_value_min=0, clip_value_max=np.inf)),
                           tf.add(weights[1],tf.clip_by_value(np.multiply(layer.get_weights(), c)[1],
                                            clip_value_min=0, clip_value_max=np.inf))])
    except IndexError:
        print('Failed')
    return layer

def calc_r(R: np.ndarray, prev_output: np.ndarray, layer, counter: int, eps: float, gamma: float, comb: bool):

    prev_output = tf.constant(prev_output)
    if comb:
        if 15 <= counter <= 16:
            eps = 0
            gamma = 0
        elif 10 <= counter < 15:
            gamma = 0
        else:
            eps = 0

    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
        layer = rho(layer, gamma)
    with tf.GradientTape() as gt:
        # forward pass / step 1
        gt.watch(prev_output)
        z = layer(prev_output)
        z = z + eps #tf.constant(0.25 * tf.reduce_mean(z**2)**.5)
        # step 2
        s = tf.divide(R, z)

        # falls LRP-0 verwendet wird, werden NaNs durch 0 ersetzt
        # das ist erlaubt, da z und R an den gleichen Stellen 0 sind
        # wir definieren also 0/0 = 0, statt NaN
        s = s.numpy()
        s[np.isnan(s)] = 0
        s = tf.constant(s)

        # step 3.1

        y = tf.reduce_sum(z*s)

    # step 3.2
    c = gt.gradient(y, prev_output)
    # step 4
    R_new = tf.constant(prev_output*c)

    return R_new
    

# Funktion für Relevance Propagation
def rel_prop(model: tf.keras.Sequential, image: np.ndarray, mask: np.ndarray, eps: float = 0, gamma: float = 0, comb: bool = False) -> np.ndarray:

    new_model = tf.keras.models.clone_model(model)
    new_model.set_weights(model.get_weights())

    layers = new_model.layers

    outputs = [image]
    for i, layer in enumerate(layers):
        outputs.append(layer(outputs[i]))

    # Anzahl der Schichten
    L = len(layers)

    # TODO: Mask durch korrektes Label definieren
    output_const = tf.constant(outputs[-1])
    output_const = output_const * mask
    R = [None]*L + [output_const]
    tmp=1
    # TODO: Vielleicht z^B-Regel für letzte Schicht anwenden --> s. Tutorial
    for l in range(0,L)[::-1]:
        layer = layers[l]
        output = outputs[l]
        R_old = R[l+1]
        if isinstance(layer, tf.keras.layers.MaxPool2D):
            layer = tf.keras.layers.AvgPool2D(layers[l].pool_size)
        # R[l] = calc_r(R[l + 1], outputs[l], layers[l])
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.AvgPool2D, tf.keras.layers.Flatten)):
            R[l] = calc_r(R_old, output, layer, l, eps, gamma, comb)
        else:
            R[l] = R_old

    relevance = np.reshape(R[0], image.shape)
    relevance = relevance.sum(axis=3)

    return relevance