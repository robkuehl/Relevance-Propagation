import time
from typing import Tuple

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import preprocess_input

from src.plotting.plot_funcs import plot_rel_prop, plot_R_evo


def run_rel_prop(model, test_images, index, regressor: bool = False,
                 z_plus_output: np.ndarray = None):
    """
    Funktion, die die Relevance Propagation startet.
    :param model: Keras Model
    :param index: Index des Inputs in Datensatz
    :param prediction: Klassifizierung des Modells
    :return: None
    """
    model = model
    image = test_images[index] * 1.0

    img = np.array([image])
    # z+
    relevance = rel_prop(model=model, image=img, regressor=regressor,
                         z_plus_output=z_plus_output)

    return relevance


def rel_prop(model: tf.keras.Sequential, image: np.ndarray, regressor: bool = False,
             z_plus_output: np.ndarray = None) -> Tuple:
    """
    Berechnet für gegebenes Model und Bild die gesamte Relevance Propagation
    :param model: Model eines KNNS
    :param image: Bild in für Modell zulässigem Format
    :param regressor: 1, wenn es sich um Regressor handelt, 0, wenn nicht
    :param z_plus_output: Relevanzwerte der Übergabeschicht
    :return: Werte der Relevance Propagation im Eingabeformat des Bildes (Ready2Plot)
    """
    # Schichten des Modells werden in Array gespeichert
    layers = model.layers

    if regressor:
        layers = [model.layers[idx] for idx in [0, 2, 4, 8]]

    # Input wird in Netz gegeben und der Output jeder Schicht wird in Array gespeichert
    outputs = [image]
    for i, layer in enumerate(layers):
        output = layer(outputs[i])

        # Auf letzten output soll keine ReLU angewandt werden
        if i < len(layers) - 1 and \
                (isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense)):
            output = tf.keras.activations.relu(output)

        outputs.append(output)

    # wenn es sich um Regressor handelt, übernehme übergebenen Output
    if regressor:
        outputs[-1] = tf.constant(z_plus_output)

    # Anzahl der Schichten
    L = len(layers)

    # Array zur Speicherung der Relevanz jeder Schicht wird initialisiert
    R = [None] * L + [outputs[-1]]

    for l in range(0, L)[::-1]:
        if isinstance(layers[l-1], tf.keras.layers.Flatten):
            break

        layer = layers[l]
        output = outputs[l]
        R_old = R[l + 1]

        # Wenn Layer zulässige Schicht ist (keine Aktivierungsfunktion o.ä., berechne R[l])
        if isinstance(layer, tf.keras.layers.Dense):
            R[l] = calc_r(R_old, output, layer)
        else:
            R[l] = R_old

    # Für letzten Schritt wird z^B verwendet
    R[0] = z_b(R[l+1], outputs[l], layers[l])

    return R


def calc_r(R: np.ndarray, prev_output: np.ndarray, layer) -> np.ndarray:
    """
    Ausführung eines Schrittes der Relevance Propagation
    :param R: Relevance der nachfolgenden Schicht (R[l+1])
    :param prev_output: Output der vorherigen Schicht
    :param layer: vorherige Schicht selbst
    :return: Relevance für vorherige Schicht
    """

    # Output der vorherigen Schicht wird in TF-Konstante transformiert
    prev_output = tf.constant(prev_output)

    # GradientTape wird aufgezeichnet
    with tf.GradientTape() as gt:
        # forward pass / step 1
        gt.watch(prev_output)

        z = forward(prev_output=prev_output, layer=layer, mode='pos')

        # step 2
        s = tf.divide(R, z)

        # NaNs werden durch 0 ersetzt -> 0/0 := 0
        s = s.numpy()
        s[np.isnan(s)] = 0
        s = tf.constant(s)

        # step 3.1
        y = tf.reduce_sum(z * s)

    # step 3.2
    c = gt.gradient(y, prev_output)

    R_new = tf.constant(prev_output * c)

    return R_new


def z_b(R: np.ndarray, prev_output: np.ndarray, layer) -> np.ndarray:
    """
    Ausführung eines Schrittes der Relevance Propagation
    :param R: Relevance der nachfolgenden Schicht (R[l+1])
    :param prev_output: Output der vorherigen Schicht
    :param layer: vorherige Schicht selbst
    :return: Relevance für vorherige Schicht
    """

    # Output der vorherigen Schicht wird in TF-Konstante transformiert
    prev_output = tf.constant(prev_output)
    prev_output = tf.cast(prev_output, tf.float32)

    low_bound = tf.constant(np.ones(prev_output.shape) * -0.5)
    low_bound = tf.cast(low_bound, tf.float32)

    high_bound = tf.constant(np.ones(prev_output.shape) * 1.5)
    high_bound = tf.cast(high_bound, tf.float32)

    # GradientTape wird aufgezeichnet
    with tf.GradientTape() as gt:
        # forward pass / step 1
        gt.watch(prev_output)
        gt.watch(low_bound)
        gt.watch(high_bound)

        z = forward(prev_output, layer, mode='normal')
        lw = forward(low_bound, layer, mode='pos')
        hw = forward(high_bound, layer, mode='neg')

        z = z - lw - hw  # + bias_term

        # step 2
        s = tf.divide(R, z)

        # NaNs werden durch 0 ersetzt -> 0/0 := 0
        s = s.numpy()
        s[np.isnan(s)] = 0
        s = tf.constant(s)

        # step 3.1
        y = tf.reduce_sum(z * s)

    # step 3.2
    c, c_low, c_high = gt.gradient(y, [prev_output, low_bound, high_bound])

    # step 4
    real = tf.constant(prev_output * c, dtype=tf.float32)
    lower = tf.constant(low_bound * c_low, dtype=tf.float32)
    higher = tf.constant(high_bound * c_high, dtype=tf.float32)
    R_new = tf.constant(real + lower + higher)

    return R_new


def forward(prev_output: tf.constant, layer: tf.keras.layers.Layer, mode: str = None) -> tf.keras.layers.Layer:
    """
    Wendet Transformation auf Gewichte der Schicht an. Momentan forward()(w) = w + c * w^+
    :param layer: Layer eines KNNs
    :param mode: String, der angibt welche Methode verwendet werden soll
                    - 'gamma': w <- w + c * w+
                    - 'pos': w <- w+
                    - 'neg': w <- w-
    :return: Layer mit transformierten Gewichten
    """

    weights = layer.get_weights().copy()

    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):

        try:

            if mode == 'pos':
                layer.set_weights([tf.clip_by_value(weights[0], clip_value_min=0, clip_value_max=np.inf)])

            elif mode == 'neg':
                layer.set_weights([tf.clip_by_value(weights[0], clip_value_min=-np.inf, clip_value_max=0)])

            elif mode is None:
                print('Falsche Eingabe für "mode" bei Aufruf von forward(.)')

        except IndexError:
            print('Failed')

    layer.activation = None

    out = layer(prev_output)

    layer.set_weights(weights)

    return out
