import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
import sys


def rho(layer: tf.keras.layers.Layer, c: int) -> tf.keras.layers.Layer:
    """
    Wendet Transformation auf Gewichte der Schicht an. Momentan rho(w) = w + c * w^+
    :param layer: Layer eines KNNs
    :param c: Faktor mit dem Positivteil der Gewichte multipliziert wird
    :return: Layer mit transformierten Gewichten
    """
    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
        try:
            weights = layer.get_weights()
            layer.set_weights([tf.add(weights[0],tf.clip_by_value(np.multiply(layer.get_weights(), c)[0],
                                                clip_value_min=0, clip_value_max=np.inf)),
                               tf.add(weights[1],tf.clip_by_value(np.multiply(layer.get_weights(), c)[1],
                                                clip_value_min=0, clip_value_max=np.inf))])
        except IndexError:
            # TODO: Logging im gesamten Projekt implementieren
            print('Failed')

    return layer


def calc_r(R: np.ndarray, prev_output: np.ndarray, layer, counter: int, eps: float, gamma: float, comb: bool) \
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
        if 15 <= counter <= 16:
            eps = 0
            gamma = 0
        elif 10 <= counter < 15:
            gamma = 0
        else:
            eps = 0

    # LRP-gamma wird angewendet
    layer = rho(layer, gamma)

    # GradientTape wird aufgezeichnet
    with tf.GradientTape() as gt:
        # forward pass / step 1
        gt.watch(prev_output)
        z = layer(prev_output)
        z = z + eps

        # step 2
        s = tf.divide(R, z)

        # NaNs werden durch 0 ersetzt -> 0/0 := 0
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
    

def rel_prop(model: tf.keras.Sequential, image: np.ndarray, mask: np.ndarray, eps: float = 0, gamma: float = 0,
             comb: bool = False) -> np.ndarray:
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

    # Kopie des Models wird angefertigt, damit Gewichte durch Funktion rho nicht für nachfolgende Anwendungen verändert
    # werden.
    new_model = tf.keras.models.clone_model(model)
    new_model.set_weights(model.get_weights())

    # Schichten des Modells werden in Array gespeichert
    layers = new_model.layers

    # Input wird in Netz gegeben und der Output jeder Schicht wird in Array gespeichert
    outputs = [image]
    for i, layer in enumerate(layers):
        outputs.append(layer(outputs[i]))

    # Anzahl der Schichten
    L = len(layers)

    # Output des Netzes wird mit Maske multipliziert, um nur zu erklärenden Output zu erhalten
    output_const = tf.constant(outputs[-1])
    output_const = output_const * mask

    # Array zur Speicherung der Relevanz jeder Schicht wird initialisiert
    R = [None]*L + [output_const]

    # Relevance Propagation wird gestartet
    # TODO: Vielleicht z^B-Regel für letzte Schicht anwenden --> s. Tutorial
    for l in range(0, L)[::-1]:
        layer = layers[l]
        output = outputs[l]
        R_old = R[l+1]

        # Wenn MaxPooling verwendet wurde, ändere in AvgPooling
        # TODO: Verfahren funktioniert scheinbar auch mit MaxPooling -> klären warum!
        if isinstance(layer, tf.keras.layers.MaxPool2D):
            layer = tf.keras.layers.AvgPool2D(layers[l].pool_size)

        # Wenn Layer zulässige Schicht ist (keine Aktivierungsfunktion o.ä., berechne R[l])
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.AvgPool2D,
                              tf.keras.layers.Flatten)):
            R[l] = calc_r(R_old, output, layer, l, eps, gamma, comb)
        else:
            R[l] = R_old

    # Reshape Output der Relevance Propagation zu Shape des Inputs
    relevance = np.reshape(R[0], image.shape)

    # Addiere Werte über Farbkanäle
    relevance = relevance.sum(axis=3)

    return relevance
