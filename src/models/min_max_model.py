#from section_3_model import plot_mnist_image
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import plotly.express as px
import numpy as np
import pickle
"""
TODO Klasse zum Speichern der benoetigten Daten:
Pro Trainingsdaten-Input wird gebraucht:
    - Input
    - Relevance Propagation Vektor der mittleren Schicht ({R_k})
    - Relevance Propagation Vektor der vorletzten Schicht ({R_l})
"""
# Define plot functions
def plot_mnist_image(image):
    fig = px.imshow(image, color_continuous_scale=px.colors.sequential.gray_r)
    fig.show()


class model_data:

    def __init__(self, dataset, relevance_propagation, save_data = False):
        self.dataset = dataset
        self.relevance_propagation = relevance_propagation
        self.mid_relevances = []
        self.high_relevances = []
        self.model_list = []
        self.save_data = save_data
   
    def set_data(self,model):
        test_input = self.dataset[0]
        test_labels = self.dataset[1]

        if self.save_data:
            #Fortschritt messen
            percentage = 5.0
            for i in range(len(test_input)):
                if i/len(test_input)* 100 > percentage:
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
        
        
    
    def fit_models(self):
        print("fitting the models and filling the list")      
        for neuron in range(100):
            print("fitting model for neuron {}".format(neuron))
            true_labels = self.load_relevances_for_neuron(neuron)
            neuron_mdl = min_max_model(true_labels, self.relevance_propagation, neuron)
            neuron_mdl.fit(self.dataset[0])
            print("true_labels shape {} ".format(true_labels.shape))
    
    
    
    def load_models(self):
        print("loading pretrained models and filling the list")      
        for neuron in range(100):
            print("loading model for neuron {}".format(neuron))
            true_labels = self.load_relevances_for_neuron(neuron)
            neuron_mdl = min_max_model(true_labels, self.relevance_propagation, neuron)
            neuron_mdl.load_mdl()
            self.model_list.append(neuron_mdl)
            print("true_labels shape {} ".format(true_labels.shape))
            
    
    def save_relevances(self):
        for neuron in range(len(self.mid_relevances[0])):
            true_labels = self.get_labels_for(neuron)
            filepath = "src/models/min_max_pickles/neuron_{}_relevances.npy".format(neuron+1)
            with open(filepath, 'wb') as file:
                np.save(file, true_labels)
            print("saved true_labels with shape {} ".format(true_labels.shape))

    def save_higher_relevances(self):
        relevances_to_save = np.asarray(self.high_relevances)
        filepath = "src/models/min_max_pickles/high_relevances.npy"
        with open(filepath, 'wb') as file:
            np.save(file, relevances_to_save)
        print("saved high_relevances with shape {} ".format(relevances_to_save.shape))
    
    def load_higher_relevances(self):
        filepath = "src/models/min_max_pickles/high_relevances.npy"
        with open(filepath, 'rb') as file:
            self.high_relevances = np.load(filepath)

    def load_relevances_for_neuron(self, neuron):
        filepath = "src/models/min_max_pickles/neuron_{}_relevances.npy".format(neuron+1)
        try:
            with open(filepath, 'rb') as file:
                true_labels = np.load(file)
                return true_labels
        except expression as identifier:
            print("exception occured while loading pickle! {}".format(identifier.name))

    def get_labels_for(self, neuron):
        val_list = []
        for i in range(len(self.mid_relevances)):
            val_list.append(self.mid_relevances[i][neuron])
        return np.asarray(val_list)
    
    def rel_prop(self,image):
        heatmap = np.zeros((28,28))
        for model in self.model_list:
            rel_map = model.rel_prop(image)
            heatmap = np.add(heatmap, rel_map)
        return heatmap

    def has_nonzero(self, image):
        for model in self.model_list:
            if model.predict(image)[0][0] > 0:
                return True
        
        return False

#END class model_data


"""
Das eigentliche min_max_model.
Bekommt das keras-model und die Dateien in model_data uebergeben
"""
class min_max_model:
    
    def __init__(self, labels, relevance_propagation, neuron_index):
        self.labels = labels
        self.relevance_propagation = relevance_propagation
        self.model = Sequential()
        self.neuron_index = neuron_index
    
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
    
    def save_mdl(self):
        filepath = 'pretrained_models/min_max_models/neuron_{}_model.h5'.format(self.neuron_index)
        self.model.save(filepath)


    def load_mdl(self):
        filepath = 'pretrained_models/min_max_models/neuron_{}_model.h5'.format(self.neuron_index)
        self.model = tf.keras.models.load_model(filepath)

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
