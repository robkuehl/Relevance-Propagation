import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.initializers import Ones
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
from src.data.get_data import get_mnist
from tensorflow.keras.optimizers import SGD
from os.path import join as pathjoin
import os
from src.rel_prop.minmax_utils import get_higher_relevances
from src.models.Binary_Mnist_Model import Montavon_Classifier
from src.rel_prop.rel_prop_min_max_adjusted import run_rel_prop
from tensorflow.keras.callbacks import ModelCheckpoint

filepath = os.path.dirname(__file__)
class Nested_Regressor():
    """
    The Nested Regressor Class wird benutzt, um die Relevanz R_j einer Schicht mit der Aktivierung/dem Input x_i der nächst höheren Schicht und den Relevanzen der nächst 
    tieferen Schicht R_l zu approximieren.
    Für den Montavon Classifier benötigen wür zu Approximation der Relevanzen im 2. Dense Layer demnach 100 Nested Regressor, jeder für 1 Neuron.
    Das Approximationsmodell des Nested Regressor ist das One-Layer Network aus Abschnitt 3 in Montavon et al. kombiniert mit einem trainierbaren Bias Term.
    Da der Bias eigene Inputdaten bekommt (R_l), bauen wir das Modell mit der Keras Functional API
    """
    
    def __init__(self, input_shape:tuple, use_bias:bool, neuron_index:int):
        """
        Args:
            input_shape (tuple): shape der Inputdaten zum trainieren des Modells, hier shape der MNIST Bilder (28,28)
            use_bias (bool): Falls True werden die higher relevances für das Training des Bias genutzt
            neuron_index (int): Index des Neurons im 2. Dense Layer des Montavon Classifiers
        """
        self.input_shape = input_shape
        self.use_bias = use_bias
        self.neuron_index = neuron_index
        self.storage_path = pathjoin(filepath, "..", "..", "models", "minmax_submodels")
        self.set_approx_model()
        print('Created nested regressor for neuron with index {}'.format(neuron_index))
        
    
    def set_approx_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(100, 
                        activation=tf.keras.activations.relu,
                        kernel_initializer=tf.keras.initializers.glorot_normal(),
                        use_bias=self.use_bias))
        sum_pooling = Dense(1, 
                            activation=tf.keras.activations.linear, 
                            use_bias=self.use_bias, 
                            kernel_initializer=Ones())
        sum_pooling.trainable=False
        model.add(sum_pooling)
        model.compile(
            optimizer = SGD(learning_rate=0.0000001),
            loss=tf.keras.losses.mse,
            metrics=[tf.keras.losses.mse]
        )
        self.model = model
        
    def fit_approx_model(self, train_images, true_relevances, higher_relevances=None):
        checkpoint = ModelCheckpoint(filepath=pathjoin(self.storage_path, "nested_regressor_{}.h5".format(self.neuron_index)))
        print('Fit model of nested regressor with neuron index {}'.format(self.neuron_index))
        self.model.fit(x=train_images, y=true_relevances, batch_size=32, epochs=300, validation_split=0.15, callbacks=[checkpoint])
        
    
    def save_model(self):
        save_model(model=self.model, filepath=pathjoin(self.storage_path, "nested_regressor_{}.h5".format(self.neuron_index)))
    
    def load_model(self):
        self.model = load_model(filepath=pathjoin(self.storage_path, "nested_regressor_{}.h5".format(self.neuron_index)))
    
        
        

class MinMaxModel():
    
    
    """
        # Für jedes Neuron im zweiten Dense Layer (100 Neuronen), erstelle einen nested_regressor
        # Lade Relevances aus zweitem Dense Layer für das entsprechende Neuron und trainiere den nested_regressor
        # Relevance Propagation für jeden nested_regressor mit z^B Regel
        # Addieren der Relevances für den Input über alle nested_regressor
        """
    
    def __init__(self, classifier:Montavon_Classifier, use_higher_rel=False):
        """Übergeordnete Klasse für das Min-Max-Modell.
           Hier führen wir den Hautpteil der Berechnug durch die definierten Methode und Klassen zusammen.
        Args:
            classifier (Montavon_Classifier): ein Objekt der Klasse Monatvon Classifier mit trainiertem Modell
            use_higher_rel (bool, optional): Entscheide ob die Relevances des 3. Dense Layers für den Bias Term der Nested Regressor verwendet werden sollen. Defaults to False.
        """
        self.use_bias = use_higher_rel
        self.classifier = classifier
        self.nested_regressors = []
        self.train_images = classifier.train_images
        self.true_relevances, self.higher_relevances, self.nr_train_images = get_higher_relevances(classifier, recalc_rel=False, use_higher_rel=use_higher_rel)
        print('Created MinMaxModel')
    
    def train_min_max(self, pretrained:bool):
        """Trainieren des Min-Max-Modells:
                1. Für jedes Neuron im 2. Dense Layer des Montavon Classifiers erstelle einen Nested Regressor mit ensprechendem Index -> 100 Modelle
                2. Trainiere die 100 Regressor Modelle mit den entsprechenden true relevances 
        Args:
            pretrained (bool): Entscheide ob bereits trainierte Modelle geladen und verwendet werden sollen
        """
        
        print('Start Training of Min-Max-Model')
        dls = [layer for layer in self.classifier.model.layers if type(layer) == Dense]
        nb_neurons = dls[1].weights[0].shape[1]
        for neuron_index in range(nb_neurons):
            nr = Nested_Regressor(
                    input_shape=self.nr_train_images[0].shape, 
                    use_bias=self.use_bias, 
                    neuron_index=neuron_index, 
                )
            self.nested_regressors.append(nr)
            
        for nr in self.nested_regressors:
            if pretrained==False:
                nr.fit_approx_model(train_images=self.nr_train_images, true_relevances=self.true_relevances[nr.neuron_index])
                #nr.save_model()
            else:
                try:
                    nr.load_model()
                except Exception:
                    nr.fit_approx_model(train_images=self.nr_train_images, true_relevances=self.true_relevances[nr.neuron_index])
                
                
    def min_max_rel_prop(self, index:int):
        """Führe die Relevance Propagation für ein Bild des Test Datensatzes mittels des trainierten Min-Max-Modells aus
        Args:
            index (int): Index des Bildes im Test Datensatz
        Returns:
            final_relevance (numpy array): Relevance für die Pixel des Inputbildes
            z_plus_relevaances (numpy array): Relevance im 2. Dense Layer des Montavon Modells erzeugt mit z+
        """
        print('Start Relevance Propagation')
        relevances = []
        pred = self.classifier.predict_test_image(index)
        
        # Führe Relevance Propagation nur dann aus, wenn das Bild label=1 hat und richtig erkannt wurde
        if pred ==1 and self.classifier.test_labels[index]==1:
            # TODO: Parallelisieren
            z_plus_relevances = run_rel_prop(
                                            model = self.classifier.model,
                                            test_images = self.classifier.test_images,
                                            test_labels = self.classifier.test_labels,
                                            classes = self.classifier.classes,
                                            index=index,
                                            prediction = self.classifier.predict_test_image(index)
                                            )[-3]
            z_plus_relevances = np.asarray(z_plus_relevances[0])
            
            # Kombinieren von z+ für tiefe Schichten mit RelProp für Aprroximationsmodelle
            # Für jedes Approximationsmodell (Nested Regressor) führe Relevance Propagation aus, wobei der Output/die zu verteilenden Relevance gegeben ist
            # durch die Werte der z_plus_relevances.
            # Der Forward Pass im Nested Regressor liefert lediglich die Regeln zur Umverteilung.
            for nr in self.nested_regressors:
                print('Starte Relevance Propagation für Nested Regressor mit Neuron Index {}'.format(nr.neuron_index))
                # TODO: Parallelisieren
                relevance = run_rel_prop(
                                        model = nr.model,
                                        test_images = self.classifier.test_images,
                                        test_labels = self.classifier.test_labels,
                                        classes = self.classifier.classes,
                                        index=index,
                                        prediction = z_plus_relevances[nr.neuron_index],
                                        regressor=True,
                                        output=z_plus_relevances[nr.neuron_index]
                                        )
                relevances.append(np.asarray(relevance))
            
            final_relevance = sum(relevances)
            return final_relevance, z_plus_relevances
        else:
            print("Model makes wrong prediction for the test image with index {}! No Relevance propagation possible!".format(index))
            return 404
            
            
            
        
        
    
        
        
    
        
    
   
        
        
        
    
            
    