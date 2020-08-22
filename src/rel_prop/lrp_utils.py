from typing import Any, Tuple
from random import shuffle

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input


def set_params(eps: float, gamma: float, no_bias: bool) -> dict:
    """
    Setzt die Parameter für alle verschiedenen Varianten
    :param eps: Epsilon
    :param gamma: Gamma
    :param no_bias: Wenn True, wird der Bias nicht benutzt
    :return: dict, das für jede Variante den Namen und die Parameter enthält
    """
    variants = ['zero', 'eps', 'gamma', 'komposition', 'plus']
    params = dict()

    for variant in variants:
        if variant == 'zero':
            params[variant] = ['LRP-0', {'eps': 0, 'gamma': 0, 'no_bias': no_bias, 'komp': False, 'z_pos': False}]

        elif variant == 'eps':
            params[variant] = [f'LRP-ε (ε={eps} * std)', {'eps': eps, 'gamma': 0, 'no_bias': no_bias, 'komp': False,
                                                          'z_pos': False}]

        elif variant == 'gamma':
            params[variant] = [f'LRP-γ (γ={gamma})', {'eps': 0, 'gamma': gamma, 'no_bias': no_bias, 'komp': False,
                                                      'z_pos': False}]

        elif variant == 'komposition':
            params[variant] = [f'LRP-Komposition', {'eps': 2*eps, 'gamma': 2*gamma, 'no_bias': no_bias, 'komp': True,
                                                    'z_pos': False}]

        elif variant == 'plus':
            params[variant] = [r'$z^+$', {'eps': 0, 'gamma': 0, 'no_bias': no_bias, 'komp': True, 'z_pos': True}]

    return params


def preprocess_for_lrp(model: tf.keras.Sequential, image: np.ndarray, mask: tf.constant) \
        -> (np.ndarray, list, tf.constant):
    """
    Das Bild wird transformiert wie beim Training. Es wird ein Forward Pass durch das Netz gemacht und der Output jeder
    Schicht wird gespeichert. Der finale Output wird mit der Maske multipliziert, um nur den Output der aktuellen
    Klassifizierung zu erhalten
    :param model: Das trainierte Neuronale Netz
    :param image: Der Input
    :param mask: Ein Array mit 1 bei der aktuellen Klasse und 0 sonst
    :return: Liste der Outputs jeder Schicht, Liste aller Schichten, finaler Output
    """
    # Originalbild wird wie beim Training des Netzes angepasst
    image = preprocess_input(image.copy())

    # Kopie des Models wird angefertigt, damit Gewichte durch Funktion forward() nicht für nachfolgende Anwendungen
    # verändert werden. Letzte Aktivierung (Sigmoid) wird gelöscht.
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

    # Output des Netzes wird mit Maske multipliziert, um nur zu erklärenden Output zu erhalten
    output_const = tf.constant(outputs[-1])
    output_const = output_const * mask

    return outputs, layers, output_const


def get_index(classifier: Any, multilabel: bool = False, n_labels: int = 2) -> int:
    """
    Gibt Index eines Bildes zurück, dass die übergebenen Parameter erfüllt
    :param classifier: Classifier mit Inputdaten
    :param multilabel: True, wenn Bild mehr als ein Label enthalten soll
    :param n_labels: Anzahl der Labels, die das Bild enthalten soll, wenn multilabel = True ist
    :return: Index des Bildes
    """
    if not multilabel:
        n_labels = 1

    indices = list(range(0, classifier.test_labels.shape[0]))
    shuffle(indices)
    for index in indices:
        if sum(classifier.test_labels[index]) == n_labels:
            labels = [classifier.classes[idx] for idx, label in enumerate(classifier.test_labels[index]) if label == 1]
            print(f'Gewähltes Bild hat Index: {index} und die Labels {labels}')
            return index
