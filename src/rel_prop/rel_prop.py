import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
import sys


def calc_r(self, r: np.ndarray, output: np.ndarray, weights: np.ndarray, eps: int = 0, beta: int = None):

    nominator = np.multiply(np.transpose(output),
                            weights)

    if beta is not None:
        if eps:
            print('+++ERROR+++')
            print('Choose either EPS or BETA, not both!')
            print('+++ERROR+++')
            sys.exit()

        zero = np.zeros(nominator.shape)
        z_pos = np.maximum(zero, nominator)
        z_neg = np.minimum(zero, nominator)

        denominator_pos = np.sum(z_pos, axis=0)
        denominator_neg = np.sum(z_neg, axis=0)

        fraction_pos = np.divide(z_pos, denominator_pos)
        fraction_neg = np.divide(z_neg, denominator_neg)

        fraction = (1 - beta) * fraction_pos + beta * fraction_neg

    else:
        denominator = np.matmul(output,
                                weights)

        if eps:
            denominator = denominator + eps * np.sign(denominator)

        fraction = np.divide(nominator, denominator)

    r_new = np.dot(fraction, r)

    return r_new
    
    
    
# Funktion für Relevance Propagation
def rel_prop(self, model: tf.keras.Sequential, image: np.ndarray, eps: float = 0, beta: float = None) -> np.ndarray:
    weights = model.get_weights()

    # Hilfsmodel zum Extrahieren der Outputs des Hidden Layers
    extractor = tf.keras.Model(inputs=model.inputs,
                                outputs=[layer.output for layer in model.layers])

    features = extractor(np.array([image]))

    outputs = [features[i] for i in range(len(list(features)))]
    
    r = []

    for i in range(0,len(outputs),-1):

        if i==len(outputs)-1:
            r_i = np.transpose(outputs[i])
            r.append(r_i)
        
        else:
            r_i = self.calc_r(r=r[-1],
                output=outputs[i],
                weights=weights[i],
                eps=eps,
                beta=beta)
            r.append(r_i)

    relevance = np.reshape(r[0], image.shape)

    return relevance