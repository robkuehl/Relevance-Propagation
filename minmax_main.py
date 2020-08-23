from src.rel_prop.min_max_rel_model import MinMaxModel, Nested_Regressor
from src.models.Binary_Mnist_Model import Montavon_Classifier
import numpy as np
from src.rel_prop.minmax_utils import get_higher_relevances
import matplotlib.pyplot as plt
import random
import time
import tensorflow as tf
from os.path import join as pathjoin
import seaborn as sns
import os
from src.plotting.plot_funcs import plot_min_max_results

dirname = os.path.dirname(__file__)

# Erstelle ein Objekt der Klasse Montavon Classifier. Das Modell wird auf dem MNIST Datemsatz trainiert zum erkennend er Klasse class_nb
mc = Montavon_Classifier(class_nb=8, load_model=True)
mc.set_data(test_size=0.25)
mc.set_model()
mc.model.summary()
mc.fit_model(epochs=300, batch_size=32)
print("Accuracy auf den Testdaten: {}".format(mc.evaluate(batch_size=32)))
#print("Accuracy auf der zu erkenneden Klasse in den Testdaten: {}".format(mc.non_trivial_accuracy()))

# Erstelle in Objekt vom Typ MinMaxModell. 
minmax = MinMaxModel(classifier=mc)

# Trainiere das MinMaxModell. Pretrained sollte auf true gesetzt werden, ansonsten werden die 100 Regressionsmodelle neu berechnet!
if tf.device('/device:cpu:0'):
    minmax.train_min_max(pretrained=True)

# Suche nach einem Bild für das Relevance Propagation durchführbar ist und führe sie mit MinMax und z+ aus. Die Ergebnisse werden im Ordner minmax_results gespeichert
nb_images = 5
for i in range(nb_images):
    while True:
        idx = random.randint(0, mc.test_images.shape[0])
        if mc.predict_test_image(idx) == 1 and mc.test_labels[idx] == 1:
            final_relevance, z_plus = minmax.min_max_rel_prop(idx)
            plot_min_max_results(mc.test_images[idx], final_relevance, z_plus, dirname, idx)
            break
    

