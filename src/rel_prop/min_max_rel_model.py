import tensorflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Ones
from tensorflow.keras.layers import Dense
import numpy as np
from src.data.get_data import get_mnist
from tensorflow.keras.optimizers import SGD

class Nested_Regressor():
    
    def __init__(self, input_shape, use_bias, neuron_index, train_images):
        self.input_shape = input_shape
        self.use_bias = use_bias
        self.neuron_index = neuron_index
        self.train_images = train_images
        
    
    def approx_model(self):
        model = Sequential()
        model.add(Dense(100, input_shape=self.input_shape, activation='relu', use_bias=self.use_bias))
        sum_pooling = Dense(1, activation='linear', use_bias=self.use_bias, kernel_initializer=Ones)
        sum_pooling.trainable=False
        model.add(sum_pooling)
        model.compile(
            optimizer = SGD(learning_rate=0.0001),
            loss='mse',
            metrcis=['loss']
        )
        self.model = model
        
    def fit_approx_model(self):
        self.model.fit(x=self.train_images, y=self.y_train, batch_size=32, epochs=300)
        
    def load_relevances(self):
        pass
        # self.y_train = relevances
    
    def load_higher_relevances(self):
        pass
    
    def save_model(self):
        # TODO: Save in /models/minmax_submodels by index
        pass
    
    def load_model(self):
        # TODO: Load from /models/minmax_submodels by index
        pass
    
        
        

class MinMaxModel():
    
    def __init__(self, montavon_model:tensorflow.keras.models.Sequential, data):
        self.montavon_model = montavon_model
        self.data = data
        self.regressors = []
    
    def minmax_rel_prop(self):
        dls = [layer for layer in self.montavon_model.layers if type(layer) == Dense]
        nb_neurons = dls[1].weights[0].shape[0]
        for neuron_index in range(0,nb_neurons):
            nr = Nested_Regressor(input_shape=self.data['train_images'][0].shape, use_bias=False, neuron_index=neuron_index, train_images=self.data['train_images'])
            self.regressors.append(nr)
        for nr in self.regressors:
            nr.fit_approx_model()
        for nr in self.regressors:
            # Führe relprop aus
            # parallelisieren
            pass
        # addiere rel_prop auf
            
            
        
        
    
        
        
    
        
    
   
        
        """
        # Für jedes Neuron im zweiten Dense Layer (100 Neuronen), erstelle einen nested_regressor
        # Lade Relevances aus zweitem Dense Layer für das entsprechende Neuron und trainiere den nested_regressor
        # Relevance Propagation für jeden nested_regressor mit z^B Regel
        # Addieren der Relevances für den Input über alle nested_regressor
        """
        
    
            
    