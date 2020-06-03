import tensorflow as tf
import numpy as np

def get_weights(self, model: tf.keras.Sequential) -> (np.ndarray, np.ndarray):
        return model.get_weights()
    
