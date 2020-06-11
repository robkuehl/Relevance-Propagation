import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
import sys


def calc_r(R: np.ndarray, prev_output: np.ndarray, layer, eps: int = 0, beta: int = None):
    rho = lambda layer: layer.set_weights()
    prev_output = tf.constant(prev_output)
    with tf.GradientTape() as gt:
        # forward pass / step 1
        gt.watch(prev_output)
        z = layer(prev_output) + 0.5
        # step 2
        s = tf.divide(R, z)
        s = tf.constant(s.numpy())
        # step 3.1

        y = tf.reduce_sum(z*s)

    # step 3.2
    c = gt.gradient(y, prev_output)
    # step 4
    R_new = tf.constant(prev_output*c)

    return R_new
    
    
    
# Funktion für Relevance Propagation
def rel_prop(model: tf.keras.Sequential, image: np.ndarray, eps: float = 0, beta: float = None) -> np.ndarray:
    layers = model.layers

    # Hilfsmodel zum Extrahieren der Outputs des Hidden Layers
    extractor = tf.keras.Model(inputs=model.inputs,
                               outputs=[layer.output for layer in model.layers])

    outputs =[image] + extractor(np.array([image]))

    # Anzahl der Schichten
    L = len(layers)

    output_const = tf.constant(outputs[-1])
    mask = np.array(output_const == np.max(output_const), dtype=np.dtype(int))

    output_const = output_const * mask
    R = [None]*L + [output_const]

    # TODO: Vielleicht z^B-Regel für letzte Schicht anwenden --> s. Tutorial
    for l in range(0,L)[::-1]:
        if isinstance(layers[l], tf.keras.layers.MaxPool2D):
            layers[l] = tf.keras.layers.AvgPool2D(layers[l].pool_size)

        # TODO: Indizes evtl. anpassen
        R[l] = calc_r(R[l+1], outputs[l], layers[l])

    relevance = np.reshape(R[0], image.shape)

    return relevance