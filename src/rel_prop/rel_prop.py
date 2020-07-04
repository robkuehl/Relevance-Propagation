import time

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import preprocess_input

from src.plotting.plot_funcs import plot_rel_prop


def run_rel_prop(classifier, eps, gamma, index, prediction):
    # index = random.randint(0, classifier.test_images.shape[0])
    model = classifier.model
    print(len(classifier.test_images))
    image = classifier.test_images[index]*1.0
    label = classifier.test_labels[index]
    dataset = 'pascal_test'

    timestamp = time.strftime('%d-%m_%Hh%M')

    label_indices = np.arange(0,len(label))[label == 1]
    for idx in label_indices:

        titles = []
        relevances = []

        persist_string = f'{dataset}_{index}_{timestamp}_class_{idx}'

        img = np.array([image])
        mask = np.zeros(len(label), dtype=np.dtype(float))
        mask[idx] = 1.

        if np.max(mask - prediction) > 1e-01:
            tmp = mask-prediction
            continue
        mask = tf.constant(mask, dtype=tf.float32)

        # LRP-0
        titles.append('LRP-0')
        relevance = rel_prop(model, img, mask)
        relevances.append(relevance)

        # LRP-eps
        titles.append(f'LRP-ε (ε={eps} * std)')
        relevance = rel_prop(model, img, mask, eps=eps)
        relevances.append(relevance)

        # LRP-gamma
        titles.append(f'LRP-γ (γ={gamma})')
        relevance = rel_prop(model, img, mask, gamma=gamma)
        relevances.append(relevance)

        # LRP-composite
        titles.append(f'LRP-Composite \neps = {2*eps}\ngamma = {2*gamma}')
        relevance = rel_prop(model, img, mask, eps=2*eps, gamma=2*gamma, comb=True)
        relevances.append(relevance)

        # z+
        titles.append('z+')
        relevance = rel_prop(model, img, mask, z_pos=True)
        relevances.append(relevance)

        relevances = tuple(zip(titles, relevances))

        plot_rel_prop(image, relevances, persist_string, False)


def rel_prop(model: tf.keras.Sequential, image: np.ndarray, mask: np.ndarray, eps: float = 0, gamma: float = 0,
             z_pos: bool = False, comb: bool = False) -> np.ndarray:
    """
    Berechnet für gegebenes Model und Bild die gesamte Relevance Propagation
    :param model: Model eines KNNS
    :param image: Bild in für Modell zulässigem Format
    :param mask: Kanonischer Basisvektor, der zu erklärendes Label indiziert
    :param eps: Epsilon für LRP-eps
    :param gamma: Gamma für LRP-gamma
    :param comb: comb: Boolean für Abfrage, ob Komposition benutzt werden soll (Anpassung der Einteilung erforderlich)
    :return: Werte der Relevance Propagation im Eingabeformat des Bildes (Ready2Plot)
    """

    image = preprocess_input(image.copy())

    # Kopie des Models wird angefertigt, damit Gewichte durch Funktion forward() nicht für nachfolgende Anwendungen verändert
    # werden. Letzte Aktivierung (Sigmoid) wird gelöscht.
    new_model = tf.keras.models.clone_model(model)
    new_model.pop()
    new_model.set_weights(model.get_weights())

    # Schichten des Modells werden in Array gespeichert
    layers = new_model.layers

    # Input wird in Netz gegeben und der Output jeder Schicht wird in Array gespeichert
    outputs = [image]
    for i, layer in enumerate(layers):
        output = layer(outputs[i])

        # Auf letzten output soll keine ReLU angewandt werden
        if i < len(layers) - 1 and \
                (isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense)):

            output = tf.keras.activations.relu(output)

        outputs.append(output)

    # Anzahl der Schichten
    L = len(layers)

    # Output des Netzes wird mit Maske multipliziert, um nur zu erklärenden Output zu erhalten
    output_const = tf.constant(outputs[-1])
    output_const = output_const * mask

    # Array zur Speicherung der Relevanz jeder Schicht wird initialisiert
    R = [None] * L + [output_const]

    # Relevance Propagation wird gestartet
    for l in range(1, L)[::-1]:
        layer = layers[l]
        output = outputs[l]
        R_old = R[l + 1]
        output_layer_bool = l == L-1

        # Wenn MaxPooling verwendet wurde, ändere in AvgPooling
        # TODO: Verfahren funktioniert scheinbar auch mit MaxPooling -> klären warum!
        if isinstance(layer, tf.keras.layers.MaxPool2D):
            layer = tf.keras.layers.AvgPool2D(layers[l].pool_size)

        # Wenn Layer zulässige Schicht ist (keine Aktivierungsfunktion o.ä., berechne R[l])
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.AvgPool2D,
                              tf.keras.layers.Flatten, tf.keras.layers.BatchNormalization)):
            R[l] = calc_r(R_old, output, layer, l, eps, gamma, z_pos, comb, output_layer_bool)
        else:
            R[l] = R_old

    # Reshape Output der Relevance Propagation zu Shape des Inputs

    R[0] = z_b(R[1], outputs[0], layers[0])

    relevance = np.reshape(R[0], image.shape)

    # Addiere Werte über Farbkanäle
    relevance = relevance.sum(axis=3)

    return relevance


def calc_r(R: np.ndarray, prev_output: np.ndarray, layer, counter: int, eps: float, gamma: float,
           z_pos: bool, comb: bool, output_layer: bool = False) \
        -> np.ndarray:
    """
    Ausführung eines Schrittes der Relevance Propagation
    :param R: Relevance der nachfolgenden Schicht (R[l+1])
    :param prev_output: Output der vorherigen Schicht
    :param layer: vorherige Schicht selbst
    :param counter: 'Nummer' des Layers, relevant bei Komposition
    :param eps: Epsilon für LRP-eps
    :param gamma: Gamma für LRP-gamma
    :param comb: Boolean für Abfrage, ob Komposition benutzt werden soll (Anpassung der Einteilung erforderlich)
    :return: Relevance für vorherige Schicht
    """

    # Output der vorherigen Schicht wird in TF-Konstante transformiert
    prev_output = tf.constant(prev_output)

    # Check, ob Komposition berechnet werden soll
    if comb:
        if isinstance(layer, tf.keras.layers.Dense):
            eps = 0
            gamma = 0
        elif 10 <= counter:
            gamma = 0
        else:
            eps = 0
    #
    # LRP-gamma wird angewendet
    #
    # GradientTape wird aufgezeichnet
    with tf.GradientTape() as gt:
        # forward pass / step 1
        gt.watch(prev_output)

        if z_pos:
            z = forward(prev_output=prev_output, layer=layer, c=gamma, mode='pos', output_layer=output_layer)
        else:
            z = forward(prev_output=prev_output, layer=layer, c=gamma, mode='gamma', output_layer=output_layer)

        # eps = eps*std
        eps = tf.constant(eps * (tf.pow(tf.pow(z, 2).numpy().mean(), 0.5)))
        z = z + eps

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


def z_b(R: np.ndarray, prev_output: np.ndarray, layer, ) -> np.ndarray:
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

    high_bound = tf.constant(np.ones(prev_output.shape) * 255.)

    # GradientTape wird aufgezeichnet
    with tf.GradientTape() as gt:
        # forward pass / step 1
        gt.watch(prev_output)
        gt.watch(low_bound)
        gt.watch(high_bound)

        z = forward(prev_output, layer)
        lw = forward(low_bound, layer, mode='pos')
        hw = forward(high_bound, layer, mode='neg')

        # eps = eps*std

        z = z - lw - hw

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
            mode: str = None, output_layer: bool = False) -> tf.keras.layers.Layer:
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
    bias_term = 0

    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
        bias_term = np.maximum(0, layer.bias.numpy())
        try:

            if mode == 'gamma':
                layer.set_weights([tf.add(weights[0], tf.clip_by_value(np.multiply(weights[0], c),
                                                                       clip_value_min=0, clip_value_max=np.inf)),
                                   tf.add(weights[1], tf.clip_by_value(np.multiply(weights[1], c),
                                                                       clip_value_min=0, clip_value_max=np.inf))])

            elif mode == 'pos':
                layer.set_weights([tf.clip_by_value(weights[0], clip_value_min=0, clip_value_max=np.inf),
                                   tf.clip_by_value(weights[1], clip_value_min=0, clip_value_max=np.inf)])

            elif mode == 'neg':
                layer.set_weights([tf.clip_by_value(weights[0], clip_value_min=-np.inf, clip_value_max=0),
                                   tf.clip_by_value(weights[1], clip_value_min=-np.inf, clip_value_max=0)])

            elif mode is None:
                pass

            else:
                print('Falsche Eingabe für "mode" bei Aufruf von forward()(.)')

        except IndexError:
            # TODO: Logging im gesamten Projekt implementieren
            print('Failed')

    layer.activation = None

    out = layer(prev_output) + bias_term
    if output_layer:
        out = tf.keras.activations.relu(out)

    layer.set_weights(weights)

    return out
