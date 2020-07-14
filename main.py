import random

import tensorflow as tf
import numpy as np

from src.cnn_eval import main_evaluate, main_load_model
from src.rel_prop.rel_prop import run_rel_prop

gpu_used = False

if gpu_used == True:
    # if GPU is used
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    
'''
Infos zum Classifier:
    model = classifier.model
    train_images = classifier.train_images
    train_labels_df = classifier.train_labels_df
'''

'''
Für main_evaluate:
    ist in cnn_eval.py in src konfiguriert
    i ist der Index für eine config aus der Liste aller Konfigurationen
'''
# classifier = main_evaluate(i=2)
'''
Für main_load_model:
    ist in cnn_eval.py in src konfiguriert
    Es muss der Pfad eines Orders angegeben werden, der durch die Methode main_evaluate angelegt wurde
'''


classifier = main_load_model('alle_Klassen_ohne_BatchNorm')
test_images = classifier.test_images
test_labels_df = classifier.test_labels_df
test_labels = test_labels_df.values
index = random.randint(0, test_labels.shape[0])

# for index in np.random.randint(0, test_labels.shape[0], 100):
for index in [449]:
#     if np.sum(test_labels[index]) != 1:
#         continue
    print(index)
    prediction = classifier.pred(index)
    run_rel_prop(classifier, eps=0.25, gamma=0.2, index=index, prediction=prediction)
