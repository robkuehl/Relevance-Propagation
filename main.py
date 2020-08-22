import random

import tensorflow as tf
import numpy as np

from src.model_evaluation import Pascal_Evaluator, Multiclass_Evaluator
from src.rel_prop.lrp import run_rel_prop

gpu_used = False

if gpu_used == True:
    # if GPU is used
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

p_e = Pascal_Evaluator()
# classifier = p_e.evaluate_config(config_nb=1)

# mce = Multiclass_Evaluator()
# mce.evaluate_config(0)


classifier = p_e.load_model('alle_Klassen_ohne_BatchNorm')

test_images = classifier.test_images
test_labels_df = classifier.test_labels_df
test_labels = test_labels_df.values
index = random.randint(0, test_labels.shape[0])
lrp_variants = ['zero', 'plus']

for index in np.random.randint(0, test_labels.shape[0], 200):
# for index in [15]:
    if np.sum(test_labels[index]) > 2:
        continue
    print(index)

    run_rel_prop(classifier, lrp_variants, index=index, eps=0.2, gamma=0.1)
