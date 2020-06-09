import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
import sys


def calc_r(R: np.ndarray, prev_output: np.ndarray, layer, eps: int = 0, beta: int = None):

    prev_output = tf.Variable(prev_output)
    with tf.GradientTape() as gt:
        # forward pass / step 1
        z = layer(prev_output)
        # step 2
        s = tf.constant(tf.divide(R, z))
        # step 3.1
        y = tf.reduce_sum(tf.tensordot(z,s))

    # step 3.2
    c = gt.gradient(y, prev_output)
    # step 4
    R_new = tf.constant(tf.tensordot(prev_output, c))

    return R_new
    
    
    
# Funktion für Relevance Propagation
def rel_prop(model: tf.keras.Sequential, image: np.ndarray, eps: float = 0, beta: float = None) -> np.ndarray:
    layers = model.layers

    # Hilfsmodel zum Extrahieren der Outputs des Hidden Layers
    extractor = tf.keras.Model(inputs=model.inputs,
                               outputs=[layer.output for layer in model.layers])

    outputs = extractor(np.array([image]))

    # Anzahl der Schichten
    L = len(layers)

    # TODO: Nur den output für relevante Klasse benutzen
    output_const = tf.constant(outputs[-1])
    R = [None]*L + output_const

    # TODO: Vielleicht z^B-Regel für letzte Schicht anwenden --> s. Tutorial
    for l in range(0,L)[::-1]:
        if isinstance(layers[l], tf.keras.layers.MaxPool2D):
            layers[l] = tf.keras.layers.AvgPool2D(layers[l].pool_size)

        # TODO: Indizes evtl. anpassen
        R[l] = calc_r(R[l+1], outputs[l], layers[l])

    relevance = np.reshape(R[0], image.shape)

    return relevance