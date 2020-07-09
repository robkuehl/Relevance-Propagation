import os
from tensorflow.keras.datasets import mnist
from src.models.section_3_model import *

"""
Testumgebung fuer Min-Max
"""

#Mnist Daten laden
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
data = [train_images, train_labels, test_images, test_labels]

# Definiere die Klasse für die wir einen binären Classifier trainieren wollen
class_nb = 8

# Trainiere das neuronale Netz des Classifiers (Code aus Vortrag 1)
cl = get_binary_cl(data=data, dataset='mnist', model_type='dense', class_nb=class_nb, epochs = 1)
model = cl.getModel()
inputs = model.inputs
#print("dtype of inputs: {}".format(inputs.dtype))
plot_images_with_rel(data[2], data[3], model, class_nb)
rp = relevance_propagation()
data[0], data[1] = cl.getBinaryData()
mdl_data = model_data(data, rp)
mdl_data.set_data(model)

 
