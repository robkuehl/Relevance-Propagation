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

def calc_r(R: np.ndarray, prev_output: np.ndarray, layer, eps: int = 0, beta: int = None):

    prev_output = tf.constant(prev_output)
    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
        layer = rho(layer, 0.25)
    with tf.GradientTape() as gt:
        # forward pass / step 1
        gt.watch(prev_output)
        z = layer(prev_output)
        z = z + tf.constant(0.25 * tf.reduce_mean(z**2)**.5)
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
def rel_prop(model: tf.keras.Sequential, image: np.ndarray, mask: np.ndarray, eps: float = 0, beta: float = None) -> np.ndarray:
    layers = model.layers

    # Hilfsmodel zum Extrahieren der Outputs des Hidden Layers
    extractor = tf.keras.Model(inputs=model.inputs,
                               outputs=[layer.output for layer in model.layers])

    outputs =[image] + extractor(np.array([image]))

    # Anzahl der Schichten
    L = len(layers)

    # TODO: Mask durch korrektes Label definieren
    output_const = tf.constant(outputs[-1])
    # mask = np.array(output_const == np.max(output_const), dtype=np.dtype(int))
    print(output_const)
    # output_const = tf.tensordot(output_const,tf.transpose(mask), axes=1)
    output_const = output_const * mask
    print(output_const)
    R = [None]*L + [output_const]

    # TODO: Vielleicht z^B-Regel für letzte Schicht anwenden --> s. Tutorial
    for l in range(0,L)[::-1]:
        layer = layers[l]
        output = outputs[l]
        R_old = R[l+1]
        if isinstance(layer, tf.keras.layers.MaxPool2D):
            layer = tf.keras.layers.AvgPool2D(layers[l].pool_size)
        # R[l] = calc_r(R[l + 1], outputs[l], layers[l])
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.AvgPool2D, tf.keras.layers.Flatten)):
            print(layer)
            R[l] = calc_r(R_old, output, layer)
        else:
            R[l] = R_old

    relevance = np.reshape(R[0], image.shape)

    return relevance