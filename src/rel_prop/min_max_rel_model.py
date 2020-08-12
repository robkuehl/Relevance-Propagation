import tensorflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Ones
from tensorflow.keras.layers import Dense
import numpy as np
from src.data.get_data import get_mnist
from tensorflow.keras.optimizers import SGD
from os.path import join as pathjoin
import os
from src.rel_prop.minmax_utils import get_higher_relevances
from src.models.Binary_Mnist_Model import Montavon_Classifier
from src.rel_prop.rel_prop import run_rel_prop

filepath = os.path.dirname(__file__)
class Nested_Regressor():
    
    def __init__(self, input_shape, use_bias, neuron_index):
        self.input_shape = input_shape
        self.use_bias = use_bias
        self.neuron_index = neuron_index
        self.storage_path = pathjoin(filepath, "..", "..", "models", "minmax_submodels")
        print('Created nested regressor for neuron with index {}'.format(neuron_index))
        
    
    def approx_model(self):
        model = Sequential()
        model.add(Dense(100, input_shape=self.input_shape, activation='relu', use_bias=self.use_bias))
        sum_pooling = Dense(1, activation='linear', use_bias=self.use_bias, kernel_initializer=Ones)
        sum_pooling.trainable=False
        model.add(sum_pooling)
        model.compile(
            optimizer = SGD(learning_rate=0.0001),
            loss='mse',
            metrics=['loss']
        )
        self.model = model
        
    def fit_approx_model(self, train_images, true_relevances, higher_relevances=None):
        print('Fit model of nested regressor with neuron index {}'.format(self.neuron_index))
        self.model.fit(x=train_images, y=true_relevances, batch_size=32, epochs=300)
        
    
    def save_model(self):
        self.model.save(pathjoin(self.storage_path, "nested_regressor_{}.h5".format(self.neuron_index)))
    
    def load_model(self):
        self.model = tensorflow.keras.models.load_model(pathjoin(self.storage_path, "nested_regressor_{}.h5".format(self.neuron_index)))
    
        
        

class MinMaxModel():
    
    def __init__(self, classifier:Montavon_Classifier, use_higher_rel=False):
        self.classifier = classifier
        self.nested_regressors = []
        self.train_images = classifier.train_images
        self.true_relevances, self.higher_relevances = get_higher_relevances(classifier, recalc=False, use_higher_rel=use_higher_rel)
        print('Created MinMaxModel')
    
    def train_min_max(self, pretrained:bool):
        print('Start Training of Min-Max-Model')
        dls = [layer for layer in self.classifier.model.layers if type(layer) == Dense]
        nb_neurons = dls[1].weights[0].shape[0]
        for neuron_index in range(0,nb_neurons):
            nr = Nested_Regressor(
                    input_shape=self.train_images[0].shape, 
                    use_bias=False, 
                    neuron_index=neuron_index, 
                )
            self.nested_regressors.append(nr)
            
        for nr in self.nested_regressors:
            if pretrained==False:
                nr.fit_approx_model(train_images=self.train_images, true_relevances=list(self.true_relevances[nr.neuron_index]))
                nr.save_model()
            else:
                nr.load_model()
                
                
    def min_max_rel_prop(self, index):
        print('Start Relevance Propagation')
        relevances = []
        # TODO: Parallelisieren
        z_plus_relevances = run_rel_prop(
                                         model = self.classifier.model,
                                         test_images = self.classifier.test_images,
                                         test_labels = self.classifier.test_labels,
                                         classes = self.classifier.classes,
                                         eps=0,
                                         gamma=0,
                                         index=index,
                                         prediction = self.classifier.predict_test_image(index)
                                         )[-3]
        
        # Kombinieren von z+ für tiefe Schichten mit RelProp für Aprroximationsmodelle
        for nr in self.nested_regressors:
            print('Starte Relevance Propagation für Nested Regressor mit Neuron Index {}'.format(nr.neuron_index))
            # TODO: Parallelisieren
            relevance = run_rel_prop(
                                    model = nr.model,
                                    test_images = self.classifier.test_images,
                                    test_labels = self.classifier.test_labels,
                                    classes = self.classifier.classes,
                                    eps=0,
                                    gamma=0,
                                    index=index,
                                    prediction = z_plus_relevances[nr.index]
                                    )
            relevances.append(relevance)
        
        final_relevance = sum(relevances)
        return final_relevance
            
            
            
        
        
    
        
        
    
        
    
   
        
        """
        # Für jedes Neuron im zweiten Dense Layer (100 Neuronen), erstelle einen nested_regressor
        # Lade Relevances aus zweitem Dense Layer für das entsprechende Neuron und trainiere den nested_regressor
        # Relevance Propagation für jeden nested_regressor mit z^B Regel
        # Addieren der Relevances für den Input über alle nested_regressor
        """
        
    
            
    