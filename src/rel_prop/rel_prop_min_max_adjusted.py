import time
from typing import Tuple

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import preprocess_input

from src.plotting.plot_funcs import plot_rel_prop, plot_R_evo


def run_rel_prop(model, test_images, test_labels, classes, index, prediction, regressor: bool = False,
                 output: np.ndarray = None):
    """
    Funktion, die die Relevance Propagation startet.
    :param model: Keras Model
    :param index: Index des Inputs in Datensatz
    :param prediction: Klassifizierung des Modells
    :return: None
    """
    model = model
    image = test_images[index] * 1.0
    label = test_labels[index]
    dataset = 'pascal_test'
    no_bias = False

    timestamp = time.strftime('%d-%m_%Hh%M')
    label_indices = np.arange(0, len(label))[label == 1]
    titles = []
    relevances = []
    evolutions_of_R = []

    R = np.zeros(100)

    # LRP wird für alle korrekten Klassifizierungen durchgeführt
    for idx in label_indices:
        correct_label = classes[idx]
        persist_string = f'{dataset}_{index}_{timestamp}_class_{idx}'

        # Maske wird erstellt, damit nur der Output der gegenwärtigen Klassifizierung genutzt wird
        img = np.array([image])
        mask = np.zeros(len(label), dtype=np.dtype(float))
        mask[idx] = 1.

        # Wenn Vorhersage zu schwach, überspringe


        mask = tf.constant(mask, dtype=tf.float32)

        # z+
        titles.append(r'$z^+$')
        relevance, relative_R_vals, R = rel_prop(model=model, image=img, mask=mask, z_pos=True, regressor=regressor, output=output)
        relevances.append(relevance)
        evolutions_of_R.append(relative_R_vals)

    return R


def rel_prop(model: tf.keras.Sequential, image: np.ndarray, mask: np.ndarray, eps: float = 0, gamma: float = 0,
             z_pos: bool = False, comb: bool = False, no_bias: bool = False, regressor: bool = False,
             output: np.ndarray = None) -> Tuple:
    """
    Berechnet für gegebenes Model und Bild die gesamte Relevance Propagation
    :param model: Model eines KNNS
    :param image: Bild in für Modell zulässigem Format
    :param mask: Kanonischer Basisvektor, der zu erklärendes Label indiziert
    :param eps: Epsilon für LRP-eps
    :param gamma: Gamma für LRP-gamma
    :param z_pos: Verwendung von z+
    :param comb: comb: Boolean für Abfrage, ob Komposition benutzt werden soll (Anpassung der Einteilung erforderlich)
    :param no_bias: Verwendung des Bias
    :return: Werte der Relevance Propagation im Eingabeformat des Bildes (Ready2Plot)
    """
    #
    # image = preprocess_input(image.copy())
    # Kopie des Models wird angefertigt, damit Gewichte durch Funktion forward() nicht für nachfolgende Anwendungen
    # verändert werden. Letzte Aktivierung (Sigmoid) wird gelöscht.

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
        outputs[-1] = output

    # Anzahl der Schichten
    L = len(layers)

    # Output des Netzes wird mit Maske multipliziert, um nur zu erklärenden Output zu erhalten
    output_const = tf.constant(outputs[-1])
    output_const = output_const * mask

    relative_R_vals = [1.]
    initial_R = output_const.numpy().sum()

    # Array zur Speicherung der Relevanz jeder Schicht wird initialisiert
    R = [None] * L + [output_const]

    # Relevance Propagation wird gestartet. Von hinten nach vorne
    if isinstance(layers[0], tf.keras.layers.Flatten):
        start = 2
    else:
        start = 1

    for l in range(start, L)[::-1]:
        layer = layers[l]
        output = outputs[l]
        R_old = R[l + 1]

        # Wenn MaxPooling verwendet wurde, ändere in AvgPooling
        if isinstance(layer, tf.keras.layers.MaxPool2D):
            layer = tf.keras.layers.AvgPool2D(layers[l].pool_size)

        # Wenn Layer zulässige Schicht ist (keine Aktivierungsfunktion o.ä., berechne R[l])
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.AvgPool2D,
                              tf.keras.layers.Flatten, tf.keras.layers.BatchNormalization)):
            R[l] = calc_r(R_old, output, layer, l, eps, gamma, z_pos, comb, no_bias)
        else:
            R[l] = R_old

        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            relative_R_vals.append(R[l].numpy().sum() / initial_R)

    # Für letzten Schritt wird z^B verwendet
    if isinstance(layers[0], tf.keras.layers.Flatten):
        R[1] = z_b(R[2], outputs[1], layers[1], regressor)
    else:
        R[0] = z_b(R[1], outputs[0], layers[0], regressor)

    relative_R_vals.append(R[1].numpy().sum() / initial_R)

    # Reshape Output der Relevance Propagation zu Shape des Inputs
    relevance = np.reshape(R[1], image.shape)

    # Addiere Werte über Farbkanäle
    # relevance = relevance.sum(axis=2)

    return relevance, relative_R_vals, R


def calc_r(R: np.ndarray, prev_output: np.ndarray, layer, counter: int, eps: float, gamma: float,
           z_pos: bool, comb: bool, no_bias: bool) \
        -> np.ndarray:
    """
    Ausführung eines Schrittes der Relevance Propagation
    :param R: Relevance der nachfolgenden Schicht (R[l+1])
    :param prev_output: Output der vorherigen Schicht
    :param layer: vorherige Schicht selbst
    :param counter: 'Nummer' des Layers, relevant bei Komposition
    :param eps: Epsilon für LRP-eps
    :param gamma: Gamma für LRP-gamma
    :param z_pos: Verwendung z+
    :param comb: Boolean für Abfrage, ob Komposition benutzt werden soll (Anpassung der Einteilung erforderlich)
    :param no_bias: Verwendung Bias
    :return: Relevance für vorherige Schicht
    """

    # Output der vorherigen Schicht wird in TF-Konstante transformiert
    prev_output = tf.constant(prev_output)

    # Check, ob Komposition berechnet werden soll
    if comb:
        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.AvgPool2D,
                              tf.keras.layers.Flatten, tf.keras.layers.BatchNormalization)):
            eps = 0
            gamma = 0
        elif 10 <= counter:
            gamma = 0
        else:
            eps = 0

    if isinstance(layer, (tf.keras.layers.AvgPool2D, tf.keras.layers.Flatten, tf.keras.layers.BatchNormalization)):
        eps = 0
        gamma = 0

    # GradientTape wird aufgezeichnet
    with tf.GradientTape() as gt:
        # forward pass / step 1
        gt.watch(prev_output)

        if z_pos:
            z = forward(prev_output=prev_output, layer=layer, c=gamma, mode='pos', no_bias=False)
        else:
            z = forward(prev_output=prev_output, layer=layer, c=gamma, mode='gamma', no_bias=no_bias)

        # eps = eps*std
        eps = tf.constant(eps * (tf.pow(tf.pow(z, 2).numpy().mean(), 0.5)))
        z = z + tf.sign(z) * eps

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


def z_b(R: np.ndarray, prev_output: np.ndarray, layer, regressor: bool) -> np.ndarray:
    """
    Ausführung eines Schrittes der Relevance Propagation
    :param R: Relevance der nachfolgenden Schicht (R[l+1])
    :param prev_output: Output der vorherigen Schicht
    :param layer: vorherige Schicht selbst
    :return: Relevance für vorherige Schicht
    """

    # Output der vorherigen Schicht wird in TF-Konstante transformiert
    # if regressor:
    #     prev_output = tf.constant(prev_output, dtype=tf.float64)
    # else:
    prev_output = tf.constant(prev_output)
    prev_output = tf.cast(prev_output, tf.float32)
    low_bound = tf.constant(np.ones(prev_output.shape) * -0.5)
    # low_bound = preprocess_input(low_bound)
    low_bound = tf.cast(low_bound, tf.float32)

    high_bound = tf.constant(np.ones(prev_output.shape) * 1.5)
    # high_bound = preprocess_input(high_bound)
    high_bound = tf.cast(high_bound, tf.float32)

    # GradientTape wird aufgezeichnet
    with tf.GradientTape() as gt:
        # forward pass / step 1
        gt.watch(prev_output)
        gt.watch(low_bound)
        gt.watch(high_bound)

        # bias_term = layer.bias.numpy()

        z = forward(prev_output, layer, c=0, mode='gamma', no_bias=True)
        lw = forward(low_bound, layer, mode='pos', no_bias=True)
        hw = forward(high_bound, layer, mode='neg', no_bias=True)

        # eps = eps*std

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


def forward(prev_output: tf.constant, layer: tf.keras.layers.Layer, c: int = 0,
            mode: str = None, no_bias: bool = False) -> tf.keras.layers.Layer:
    """
    Wendet Transformation auf Gewichte der Schicht an. Momentan forward()(w) = w + c * w^+
    :param layer: Layer eines KNNs
    :param c: Faktor mit dem Positivteil der Gewichte multipliziert wird
    :param mode: String, der angibt welche Methode verwendet werden soll
                    - 'gamma': w <- w + c * w+
                    - 'pos': w <- w+
                    - 'neg': w <- w-
    :return: Layer mit transformierten Gewichten
    """

    weights = layer.get_weights().copy()

    # falls positiver Bias auftritt, behandele diesen wie zusätzliche Neuronen -> max(0, b_i) forward pass hinzufügen

    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):

        try:

            if mode == 'gamma':
                layer.set_weights([tf.add(weights[0], tf.clip_by_value(np.multiply(weights[0], c),
                                                                       clip_value_min=0, clip_value_max=np.inf))])

            elif mode == 'pos':
                layer.set_weights([tf.clip_by_value(weights[0], clip_value_min=0, clip_value_max=np.inf)])

            elif mode == 'neg':
                layer.set_weights([tf.clip_by_value(weights[0], clip_value_min=-np.inf, clip_value_max=0)])

            elif mode is None:
                pass

            else:
                print('Falsche Eingabe für "mode" bei Aufruf von forward(.)')

        except IndexError:
            # TODO: Logging im gesamten Projekt implementieren
            print('Failed')

    layer.activation = None

    out = layer(prev_output)

    layer.set_weights(weights)

    return out
