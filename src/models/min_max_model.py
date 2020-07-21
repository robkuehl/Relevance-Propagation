from src.models.min_max_utils import plot_mnist_image
from src.models.min_max_utils import plot_rel
from src.models.min_max_utils import getMnistModel
from src.models.min_max_utils import relevance_propagation
import tensorflow as tf
import random
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import plotly.express as px
import numpy as np

"""
    Klasse zur Speicherung des Min-Max-Modells:
    Pro Neuron, dass durch Hilfsmodell approximiert wird, wird gebraucht:
    - Alle Inputgrafiken
    - Relevanz Werte dieses Neurons in der Mittleren Schicht
    - Relevance Propagation Vektoren der vorletzten Schicht ({R_l}) für alle Inputs
"""
class min_max:

    def __init__(self, dataset, relevance_propagation, save_data = False):
        self.dataset = dataset
        self.relevance_propagation = relevance_propagation
        #ggf. weglassen?
        self.mid_relevances = []
        self.high_relevances = []
        self.model_list = []
        self.save_data = save_data

    """
        Die Flag save_data gibt an, ob die hoeheren Relevanzen neu extrahiert werden sollen (Modellparameter geaendert)
        oder die bereits bestehenden eingelesen werden sollen.
    """
    def set_data(self,model):
        test_input = self.dataset[0]
        test_labels = self.dataset[1]

        if self.save_data:
            #Debug-Zwecke: ggf. nur einen Teil der Daten extrahieren (bei len(test_input) wird der komplette Datensatz extrahiert)
            data_to_extract = len(test_input)
            #Fortschritt messen
            percentage = 5.0
            for i in range(data_to_extract):
                if i/data_to_extract * 100 > percentage:
                    print("{} percent done".format(percentage))
                    percentage +=5.0
                #Extrahiere die Relevanzen, die Notwendig sind um die models zu fitten
                high_relevance, mid_relevance = self.relevance_propagation.get_higher_relevances(model,test_input[i])
                self.mid_relevances.append(mid_relevance)
                self.high_relevances.append(high_relevance)
            self.high_relevances = np.asarray(self.high_relevances)
            self.save_higher_relevances()
            self.save_relevances()
        else:
            self.load_higher_relevances()
    
    
    """
        Integrierte Relevance-Propagation: Fuehre fuer jedes gefittete Hilfsmodell eine eigene LRP aus und addiere die Ergebnisse
    """
    def rel_prop(self,image):
        heatmap = np.zeros((28,28))
        for model in self.model_list:
            rel_map = model.rel_prop(image)
            heatmap = np.add(heatmap, rel_map)
        return heatmap        
        
    """
        Entweder alle Hilfsmodelle fitten
    """
    def fit_models(self):
        print("fitting the models and filling the list")      
        for neuron in range(100):
            print("fitting model for neuron {}".format(neuron))
            true_labels = self.load_relevances_for_neuron(neuron)
            neuron_mdl = approx_model(true_labels, self.relevance_propagation, neuron)
            neuron_mdl.fit(self.dataset[0])
            print("true_labels shape {} ".format(true_labels.shape))
    
    
    """
        Oder alle Hilfsmodelle laden
    """
    def load_models(self):
        print("loading pretrained models and filling the list")      
        for neuron in range(100):
            print("loading model for neuron {}".format(neuron))
            true_labels = self.load_relevances_for_neuron(neuron)
            neuron_mdl = approx_model(true_labels, self.relevance_propagation, neuron)
            neuron_mdl.load_mdl()
            self.model_list.append(neuron_mdl)
            
    """
        Hilfsmethode, nach dem extrahieren werden die zu approximierenden Relevanzen in einzelnen Datensaetzen gespeichert
    """
    def save_relevances(self):
        for neuron in range(len(self.mid_relevances[0])):
            true_labels = self.get_labels_for(neuron)
            filepath = "src/models/min_max_pickles/neuron_{}_relevances.npy".format(neuron+1)
            with open(filepath, 'wb') as file:
                np.save(file, true_labels)

    """
        Hilfsmethode, nach dem extrahieren werden die Relevanzen aus der tieferen Schicht (R_l) in einem grossen Datensatz gespeichert
    """
    def save_higher_relevances(self):
        relevances_to_save = np.asarray(self.high_relevances)
        filepath = "src/models/min_max_pickles/high_relevances.npy"
        with open(filepath, 'wb') as file:
            np.save(file, relevances_to_save)
        print("saved high_relevances with shape {} ".format(relevances_to_save.shape))

    """
        Hilfsmethode, lade die Relevanzen aus der tieferen Schicht (R_l)
    """   
    def load_higher_relevances(self):
        filepath = "src/models/min_max_pickles/high_relevances.npy"
        with open(filepath, 'rb') as file:
            self.high_relevances = np.load(filepath)
    """
        Hilfsmethode, lade die zu approximierenden Relevanzen fuer das uebergebene neuron
    """
    def load_relevances_for_neuron(self, neuron):
        filepath = "src/models/min_max_pickles/neuron_{}_relevances.npy".format(neuron+1)
        try:
            with open(filepath, 'rb') as file:
                true_labels = np.load(file)
                return true_labels
        except expression as identifier:
            print("exception occured while loading pickle! {}".format(identifier.name))

    """
        Hilfsmethode: Speichere die Relevanzen, die das min-max-modell fuer 'neuron' approximieren soll.
    """
    def get_labels_for(self, neuron):
        val_list = []
        for i in range(len(self.mid_relevances)):
            val_list.append(self.mid_relevances[i][neuron])
        return np.asarray(val_list)

    """
        Hilfsmethode, um schoene Bildchen zu finden:
        Ueberpruefe, ob mindestens eines der Hilfsmodelle eine positive Prediction liefert
        Andernfalls ist die Heatmap neutral grau
    """
    def has_nonzero(self, image):
        for model in self.model_list:
            if model.predict(image)[0][0] > 0:
                return True
        
        return False

#END class model_data


"""
    Hilfsmodell der Form aus dem Paper: Dient zur Approximierung der Relevanzen des zugrundeliegenden Modells
"""
class approx_model:
    
    #Wird mit eigener relevance_propagation instanz ausgestattet
    def __init__(self, labels, relevance_propagation, neuron_index):
        self.labels = labels
        self.relevance_propagation = relevance_propagation
        self.model = Sequential()
        self.neuron_index = neuron_index
    
    #TODO Hier dran arbeiten: Das Modell lernt nix!
    def fit(self, images):
        self.initializeModel()
        self.model.compile(optimizer=SGD(learning_rate = 0.0001),
                loss=tf.keras.losses.MeanAbsoluteError(),
                metrics=['acc'])
        
        self.model.fit(
                images,
                self.labels,
                epochs=2,
                batch_size=20,
                verbose=2
        )
        self.save_mdl()
    
    """
        Hilfsmethoden, sollten selbsterklaerend sein
    """
    
    def save_mdl(self):
        filepath = 'pretrained_models/min_max_models/neuron_{}_model.h5'.format(self.neuron_index)
        self.model.save(filepath)


    def load_mdl(self):
        filepath = 'pretrained_models/min_max_models/neuron_{}_model.h5'.format(self.neuron_index)
        self.model = tf.keras.models.load_model(filepath)

    #Das zugrundeliegende model wird initialisiert
    #TODO HIER DIE v_lj integrieren! -> Skip connections oder mit functional API arbeiten -> multiple input
    def initializeModel(self):
        ones_initializer = tf.keras.initializers.Ones()
        self.model.add(Flatten(input_shape=(28,28)))
        self.model.add(Dense(20, activation='relu', use_bias = False))
        #Kernel initializer sorgt dafuer, dass die Gewichtsmatrix die geforderte Pooling Operation realisiert
        sum_pooling = Dense(100, activation = 'sigmoid', use_bias = False, kernel_initializer = ones_initializer)
        #Gewichte sollen nicht veraendert werden
        sum_pooling.trainable=False
        self.model.add(sum_pooling)
    
    def rel_prop(self, image):
        return self.relevance_propagation.rel_prop(self.model, image)

    def predict(self, image):
        return self.model.predict(image.reshape(1,28,28))
#END class approx_model


############################################################################################################
########### METHODEN ZUR VISUALISIERUNG ####################################################################
############################################################################################################
""" 
    Plotte 3 Grafiken: Das mnist Bild, Min-Max-Modell-LRP und Standard LRP-0 Regel (z-Regel)
"""
def plot_rel_prop_comparison(image, model, min_max_mdl):
    plot_mnist_image(image)               
    heatmap_z = relevance_propagation().rel_prop(model, image)
    heatmap_z_min_max = min_max_mdl.rel_prop(image)
    plot_rel(heatmap_z)
    plot_rel(heatmap_z_min_max)


########################################################################################################
######################## CODE AUS DEM ERSTEN VORTRAG, (TRAINING FREE ANSATZ) ###########################
########################################################################################################


#Funktion zum Erzeugen eines binaeren Classifiers, der nur class_nb erkennt
def get_binary_cl(data, dataset, model_type, class_nb, epochs=10, batch_size=20):
    cl = binary_classifier(model_type=model_type, dataset=dataset, class_nb=class_nb)
    cl.set_data(data)
    cl.set_model()
    cl.fit_model(epochs, batch_size)

    return cl



""" 
    Klasse aus dem ersten Vortrag, wird diesmal mit dem Sec. III Model initialisiert
"""
class binary_classifier:
    
    def __init__(self, model_type, dataset, class_nb):
        self.model_type = model_type
        assert(type(self.model_type)==str)
        self.dataset = dataset
        self.class_nb = class_nb
        

    def set_data(self, data):
        
        train_images = data[0]
        train_labels =  data[1]
        self.test_images = data[2]
        self.test_labels = data[3]
        
        #make_binary_data
        train_labels = (train_labels==self.class_nb).astype(int)
        self.test_labels = (self.test_labels==self.class_nb).astype(int)
        
         # reduce train dataset
        one_indices = [i for i in range(train_labels.shape[0]) if train_labels[i]==1]
        zero_indices = [i for i in range(train_labels.shape[0]) if train_labels[i]==0]
        sampling = random.choices(zero_indices, k=3*len(one_indices))
        train_indices = one_indices + sampling
        print("Number of train indices: ", len(train_indices))
        self.train_images = np.asarray([train_images[i] for i in train_indices])
        print(self.train_images.shape)
        self.train_labels = np.asarray([train_labels[i] for i in train_indices])
        
    def getBinaryLabels(self):
        return self.train_labels, self.test_labels
        

    def set_model(self):
        
        if self.dataset == 'mnist':
            input_shape=(28,28)
        elif self.dataset == 'cifar10':
            input_shape=(32,32,3)
        

        if self.model_type == "dense":
            
            """AENDERUNG: BENUTZE MODEL AUS SEC III VON MONTAVON ET AL."""
            model = getMnistModel()
            """AENDERUNG: BENUTZE HINGE LOSS UND SGD VERFAHREN"""
            model.compile(loss=tf.keras.losses.Hinge(),
                        optimizer=SGD(learning_rate = 0.0001),
                        metrics=['acc'])           
            

        if self.model_type == "load_dense":
            filepath = 'pretrained_models/montavon_mnist_model.h5'
            model = tf.keras.models.load_model(filepath)

        model.summary()

        self.model = model


    def fit_model(self, epochs: int, batch_size: int):
        if self.model_type == "dense":
            #Enable/Disable gpu support (linux)
            forceDeviceUse = False
            deviceToUse = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
            if forceDeviceUse:
                with tf.device(deviceToUse):
                    self.model.fit(
                        self.train_images,
                        self.train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(self.test_images, self.test_labels),
                        verbose=2
                    )
            else:
                #seems to work well with windows: let tensorflow decide which device to use
                self.model.fit(
                    self.train_images,
                    self.train_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(self.test_images, self.test_labels),
                    verbose=2
                )

            filepath = 'montavon_mnist_model.h5'
            """AENDERUNG: SPEICHERE MODEL."""
            self.model.save(filepath)
        

    def predict(self, image):
        pred = int(self.model.predict(np.array([image]))[0][0])
        return pred
    
    #Nicht sinnvoll bei aktueller Modellarchitektur?
    def non_trivial_accuracy(self):
        answers = []
        for i in range(len(list(self.test_labels))):
            if self.test_labels[i]==1:
                answers.append(int(self.model.predict(np.array([self.test_images[i]]))[0][0]))
                
        return sum(answers)/len(answers)
    
    def evaluate(self, batch_size):
        _ , acc = self.model.evaluate(self.test_images, self.test_labels,
                                batch_size=batch_size)
        return acc

    def getModel(self):
        return self.model
#END class binary_classifier