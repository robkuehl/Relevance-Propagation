import random

import tensorflow as tf
import numpy as np

from src.model_evaluation import Pascal_Evaluator, Multiclass_Evaluator
from src.rel_prop.rel_prop import run_rel_prop

gpu_used = True

if gpu_used == True:
    # if GPU is used
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    

# p_e = Pascal_Evaluator()
# classifier = p_e.evaluate_config(config_nb=1)

mce = Multiclass_Evaluator()
mce.evaluate_config(0)



