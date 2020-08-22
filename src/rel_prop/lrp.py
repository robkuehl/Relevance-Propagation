import time
from typing import Tuple, Any

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import preprocess_input

from src.plotting.plot_funcs import plot_rel_prop, plot_R_evo
from src.rel_prop.lrp_utils import set_params, preprocess_for_lrp


def run_rel_prop(classifier: Any, lrp_variants: [str], index: int, eps: float = 0., gamma: float = 0.,
                 show_viz: bool = True, show_conservation_plots: bool = False) -> None:
    """
    Funktion, die die Relevance Propagation startet.
    :param classifier: Classifier, der Daten und das trainierte Netz enthält
    :param lrp_variants: Liste von gewünschten LRP-Varianten, choose from: 'zero', 'eps', 'gamma', 'komposition', 'plus'
    :param eps: Parameter für LRP-epsilon
    :param gamma: Parameter für LRP-gamma
    :param index: Index des Inputs in Datensatz
    :param show_viz: Soll Visualisierung angezeigt werden?
    :param show_conservation_plots: Soll Verlauf der relativen Relevanzwerte gezeigt werden? (Plots s. Abb. 13)
    :return: None
    """
    model = classifier.model
    image = classifier.test_images[index]*1.0
    label = classifier.test_labels[index]
    prediction = classifier.pred(index)
    classes = classifier.classes
    dataset = 'pascal_test'
    no_bias = False

    params = set_params(eps, gamma, no_bias)

    timestamp = time.strftime('%d-%m_%Hh%M')

    # sucht die Indices der korrekten Labels
    label_indices = np.arange(0, len(label))[label == 1]

    # LRP wird für alle korrekten Klassifizierungen durchgeführt
    for idx in label_indices:
        titles = []
        relevances = []
        evolutions_of_R = []
        correct_label = classes[idx]
        persist_string = f'{dataset}_{index}_{timestamp}_class_{idx}'

        # Maske wird erstellt, damit nur der Output der gegenwärtigen Klassifizierung genutzt wird
        img = np.array([image])
        mask = np.zeros(len(label), dtype=np.dtype(float))
        mask[idx] = 1.
        mask = tf.constant(mask, dtype=tf.float32)

        # Wenn Vorhersage zu schwach, überspringe
        if np.max(mask - prediction) > 2e-01:
            continue

        for variant in lrp_variants:
            # LRP wird angewendet
            titles.append(params[variant][0])
            relevance, relative_R_vals = rel_prop(model=model, image=img, mask=mask, **params[variant][1])
            relevances.append(relevance)
            evolutions_of_R.append(relative_R_vals)

        relevances = tuple(zip(titles, relevances))
        evolutions_of_R = tuple(zip(titles, evolutions_of_R))

        # plottet Verlauf der Summe über alle R
        plot_R_evo(evolutions_of_R, persist_string, show_conservation_plots)

        # plottet die Visualisierung
        plot_rel_prop(image, correct_label, relevances, persist_string, show_viz)


def rel_prop(model: tf.keras.Sequential, image: np.ndarray, mask: np.ndarray, eps: float, gamma: float,
             z_pos: bool, komp: bool, no_bias: bool) -> Tuple:
    """
    Berechnet für gegebenes Model und Bild die gesamte Relevance Propagation
    :param model: Model eines KNNS
    :param image: Bild in für Modell zulässigem Format
    :param mask: Kanonischer Basisvektor, der zu erklärendes Label indiziert
    :param eps: Epsilon für LRP-eps
    :param gamma: Gamma für LRP-gamma
    :param z_pos: Verwendung von z+
    :param komp: komp: Boolean für Abfrage, ob Komposition benutzt werden soll (Anpassung der Einteilung erforderlich)
    :param no_bias: Verwendung des Bias
    :return: Werte der Relevance Propagation im Eingabeformat des Bildes (Ready2Plot)
    """
    outputs, layers, output_const = preprocess_for_lrp(model, image, mask)

    # Anzahl der Schichten
    L = len(layers)
    relative_R_vals = [1.]
    initial_R = output_const.numpy().sum()

    # Array zur Speicherung der Relevanz jeder Schicht wird initialisiert
    R = [None] * L + [output_const]

    # Relevance Propagation wird gestartet. Von output Richtung input
    for l in range(1, L)[::-1]:
        layer = layers[l]
        output = outputs[l]
        R_old = R[l + 1]

        # Wenn MaxPooling verwendet wurde, ändere in AvgPooling
        if isinstance(layer, tf.keras.layers.MaxPool2D):
            layer = tf.keras.layers.AvgPool2D(layers[l].pool_size)

        # Wenn Layer zulässige Schicht ist (keine Aktivierungsfunktion o.ä., berechne R[l])
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.AvgPool2D,
                              tf.keras.layers.Flatten, tf.keras.layers.BatchNormalization)):

            R[l] = calc_r(R_old, output, layer, l, eps, gamma, z_pos, komp, no_bias)

        else:

            R[l] = R_old

        # Berechne Summe der Relevanzwerte relativ zum Output
        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            relative_R_vals.append(R[l].numpy().sum() / initial_R)

    # Für letzten Schritt wird z^B verwendet
    R[0] = z_b(R[1], outputs[0], layers[0])
    relative_R_vals.append(R[0].numpy().sum() / initial_R)

    # Reshape Output der Relevance Propagation zu Shape des Inputs
    relevance = np.reshape(R[0], image.shape)

    # Addiere Werte über Farbkanäle
    relevance = relevance.sum(axis=3)

    return relevance, relative_R_vals


def calc_r(R: np.ndarray, prev_output: np.ndarray, layer, counter: int, eps: float, gamma: float,
           z_pos: bool, komp: bool, no_bias: bool) \
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
    :param komp: Boolean für Abfrage, ob Komposition benutzt werden soll (Anpassung der Einteilung erforderlich)
    :param no_bias: Verwendung Bias
    :return: Relevance für vorherige Schicht
    """

    # Output der vorherigen Schicht wird in TF-Konstante transformiert
    prev_output = tf.constant(prev_output)

    # Check, ob Komposition berechnet werden soll
    if komp:
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
            z = forward(prev_output=prev_output, layer=layer, c=gamma, mode='pos', no_bias=no_bias)
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

    R_new = tf.constant(prev_output*c)

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

    low_bound = tf.constant(np.zeros(prev_output.shape))
    low_bound = preprocess_input(low_bound)

    high_bound = tf.constant(np.ones(prev_output.shape) * 255.)
    high_bound = preprocess_input(high_bound)
    # GradientTape wird aufgezeichnet
    with tf.GradientTape() as gt:
        # forward pass / step 1
        gt.watch(prev_output)
        gt.watch(low_bound)
        gt.watch(high_bound)

        bias_term = layer.bias.numpy()

        # Bias soll nicht Einschränkung betroffen sein
        z = forward(prev_output, layer, c=0, mode='gamma', no_bias=True)
        lw = forward(low_bound, layer, mode='pos', no_bias=True)
        hw = forward(high_bound, layer, mode='neg', no_bias=True)

        z = z - lw - hw + bias_term

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
    R_new = tf.constant(prev_output * c + low_bound * c_low + high_bound * c_high)

    return R_new


def forward(prev_output: tf.constant, layer: tf.keras.layers.Layer, c: int = 0,
            mode: str = None, no_bias: bool = False) -> tf.keras.layers.Layer:
    """
    Wendet Transformation auf Gewichte der Schicht für forward pass an
    :param layer: Layer eines KNNs
    :param c: Faktor mit dem Positivteil der Gewichte multipliziert wird
    :param mode: String, der angibt welche Methode verwendet werden soll
                    - 'gamma': w <- w + c * w+
                    - 'pos': w <- w+
                    - 'neg': w <- w-
    :return: Layer mit transformierten Gewichten
    """
    # Gewichte werden kopiert
    weights = layer.get_weights().copy()

    # Transformation soll nicht auf Hilfslayer angewendet werden
    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):

        # Wenn Bias nicht verwendet wird, setze ihn auf 0
        if no_bias:
            weights[1] = weights[1]*0.

        # Positive Gewichte werden mit Faktor (1 + c) vergrößert
        if mode == 'gamma':
            layer.set_weights([tf.add(weights[0], tf.clip_by_value(np.multiply(weights[0], c),
                                                                   clip_value_min=0, clip_value_max=np.inf)),
                               tf.add(weights[1], tf.clip_by_value(np.multiply(weights[1], c),
                                                                   clip_value_min=0, clip_value_max=np.inf))])

        # Nur positive Gewichte werden benutzt. Positive Biaswerte werden beibehalten und theoretisch als weieres
        # Neuron angesehen, um b ≤ 0 zu schützen
        elif mode == 'pos':
            layer.set_weights([tf.clip_by_value(weights[0], clip_value_min=0, clip_value_max=np.inf),
                               tf.clip_by_value(weights[1], clip_value_min=0, clip_value_max=np.inf)])

        # Nur negative Gewichte werden benutzt
        elif mode == 'neg':
            layer.set_weights([tf.clip_by_value(weights[0], clip_value_min=-np.inf, clip_value_max=0),
                               tf.clip_by_value(weights[1], clip_value_min=-np.inf, clip_value_max=0)])

        elif mode is None:
            pass

        else:
            print('Falsche Eingabe für "mode" bei Aufruf von forward(.)')

    # Gesamtinput soll berechnet, dazu darf die Aktivierung noch nicht angewendet werden
    layer.activation = None

    # Forward pass wird durchgeführt
    out = layer(prev_output)

    # Ursprungsgewichte werden wieder gesetzt
    layer.set_weights(weights)

    return out
