import tensorflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Ones
from tensorflow.keras.layers import Dense
import numpy as np
from src.data.get_data import get_mnist

class Nested_Regressor():
    
    def __init__(self, input_shape, use_bias, neuron_index, data):
        self.input_shape = input_shape
        self.use_bias = use_bias
        self.neuron_index = neuron_index
        self.data = data
        
    
    def approx_model(self):
        model = Sequential()
        model.add(Dense(100, input_shape=self.data['train_images'][0].shape, activation='relu', use_bias=self.use_bias))
        sum_pooling = Dense(1, activation='linear', use_bias=self.use_bias, kernel_initializer=Ones)
        sum_pooling.trainable=False
        model.add(sum_pooling)
        self.model = model
        
    def load_relevances():
        
    
    def load_higher_relevances():
        
        
        

class MinMaxModel():
    
    def __init__(self, classifier:tensorflow.keras.model, data):
        self.classifier = classifier
        self.data = data
        
    for 
        
        
    
        
        
    
        
    
   
        
        """
        # Für jedes Neuron im zweiten Dense Layer (100 Neuronen), erstelle einen nested_regressor
        # Lade Relevances aus zweitem Dense Layer für das entsprechende Neuron und trainiere den nested_regressor
        # Relevance Propagation für jeden nested_regressor mit z^B Regel
        # Addieren der Relevances für den Input über alle nested_regressor
        """
        
    
            
    