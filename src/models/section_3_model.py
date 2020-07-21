from src.models.min_max_model import model_data
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import random
import plotly.express as px
import sys
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
import numpy as np
import os


"""
Definiert eine Matrix, die inputDim mittels Sum Pooling auf outputDim reduziert
"""
def getSumPoolingWeights(inputDim = 400, outputDim = 100):
    #Bestimme die Anzahl an Neuronen, die auf ein Outputneuron summiert werden
    pool_ratio = int(inputDim /outputDim)
    #TODO Fehler werfen, falls kein int!!
    #Liste zum Speichern der Zeilen der Matrix
    row_list = []
    for row in range(outputDim):
        this_row = np.zeros(inputDim)
        #Zeile soll nur <pool_ratio> viele 1en an der richtigen Position haben
        this_row[pool_ratio * row:pool_ratio*row+pool_ratio]=1.0
        row_list.append(this_row)
    #Gewichtsmatrix setzt sich zusammen aus den erzeugten Zeilen
    weight_matrix = np.asarray(row_list)    
    return weight_matrix


"""
Das Modell aus Sec III in Montavon et al
Besteht aus: 
Flattened Input
Dense Layer mit 400 Detektionsneuronen, relu aktivierung
Pooling Operation: 400 neuronen jeweils 4 summieren -> 100 neuronen Output
Dense Layer mit 400 Detektionsneuronen, relu aktivierung
Globales sum-pooling zu einem outputneuron. Output soll ungefähr 1 sein, falls Zahl erkannt wurde, 0 sonst
""" 
def getSection3Model(input_shape = (28,28)):
    #Eingebaute Funktion, die die Uebergangsmatrix mit 1en initialisiert.
    ones_initializer = tf.keras.initializers.Ones()
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(400, activation='relu', use_bias = False))
    #Kernel initializer sorgt dafuer, dass die Gewichtsmatrix die geforderte Pooling Operation realisiert
    custom_pooling = Dense(100, activation = 'relu', use_bias = False, kernel_initializer = ones_initializer)
    #Gewichte sollen nicht veraendert werden
    custom_pooling.trainable=False
    model.add(custom_pooling)
    model.add(Dense(400, activation='relu', use_bias = False))
    #Gleiches wie oben, kernel wird mit 1en initialisiert und nicht trainierbar -> sum-pooling
    sum_pooling = Dense(1, activation = 'relu', use_bias = False, kernel_initializer = ones_initializer)
    sum_pooling.trainable = False
    model.add(sum_pooling)
    #Uebergangsmatrix bei sum_pooling aendern:
    list_of_weights = [np.transpose(getSumPoolingWeights())]
    #print("list of weights [0] shape: {}, [1] shape {}".format(list_of_weights[0].shape, list_of_weights[1].shape))
    model.layers[2].set_weights(list_of_weights)
    return model



########################################################################################################
######################## CODE AUS DEM ERSTEN VORTRAG, GGF. UEBERARBEITET ###############################
########################################################################################################

def plot_rel(image):
    imageArray = np.asarray(image)
    #get min and max value and define the bound for heatmap
    min_val = np.amin(imageArray)
    max_val = np.amax(imageArray)
    bound = np.amax(np.array([-1* min_val, max_val]))
    fig = px.imshow(image, color_continuous_scale=px.colors.sequential.Cividis, zmin=-1*bound, zmax=bound)
    fig.show()

# Define plot functions
def plot_mnist_image(image):
    fig = px.imshow(image, color_continuous_scale=px.colors.sequential.gray_r)
    fig.show()


#Funktion zum Erzeugen eines binaeren Classifiers, der nur class_nb erkennt
def get_binary_cl(data, dataset, model_type, class_nb, epochs=10, batch_size=20):
    cl = binary_classifier(model_type=model_type, dataset=dataset, class_nb=class_nb)
    cl.set_data(data)
    cl.set_model()
    cl.fit_model(epochs, batch_size)

    #print("Model Accuracy: {}".format(cl.evaluate(10)))
    #print("Model Accuracy for images with label {} : {}".format(class_nb, cl.non_trivial_accuracy()))
    # TODO Weg zum Speichern und Laden des model finden -> Custom kernel initializer bereitet Probleme
    return cl

def plot_rel_prop(image, model):
    plot_mnist_image(image)               
    rel = relevance_propagation().rel_prop(model, image)
    plot_rel(rel)
    print(test_labels[i])   

#Funktion zum Plotten der Relevance_propagation
def plot_images_with_rel(test_images, test_labels, model, nb_class):
    # stelle ein Objekt der Klasse Relevance Propagation
    rp = relevance_propagation()
    num_images = 1
    # Führe Relevance Propagation für die ersten <num_images> Bilder der Klasse nb_class aus, die der Classifier korrekt erkennt
    j=0
    i=0
    while j<num_images:
        if test_labels[i]==nb_class:
            prediction = model.predict(np.array([test_images[i]]))
       

            if prediction[0][0]>=0.5:
                print("predicted :{} on image:".format(prediction))
                plot_mnist_image(test_images[i])
                j+=1
                image = test_images[i]
                rel = rp.rel_prop(model, image)
                plot_rel(rel)
                print(test_labels[i])
        i+=1


class relevance_propagation:
    
    def get_weights(self, model: tf.keras.Sequential) -> (np.ndarray, np.ndarray):
        #TODO return array for variable number of layers

        first_weights = model.weights[0].numpy()
        second_weights = model.weights[1].numpy()
     
        return first_weights, second_weights


    def calc_r(self, r: np.ndarray, output: np.ndarray, weights: np.ndarray, eps: int = 0, beta: int = None):

        #print("calling calc_r with shapes: output: {}, weights: {}, beta {} and eps {}".format(output.shape, weights.shape, beta, eps))
        nominator = np.multiply(np.transpose(output),
                                weights)
        #print("neuron values: {}".format(output))
        if beta is not None:
            if eps:
                print('+++ERROR+++')
                print('Choose either EPS or BETA, not both!')
                print('+++ERROR+++')
                sys.exit()

            zero = np.zeros(nominator.shape)
            z_pos = np.maximum(zero, nominator)
            z_neg = np.minimum(zero, nominator)

            denominator_pos = np.sum(z_pos, axis=0)
            denominator_neg = np.sum(z_neg, axis=0)

            fraction_pos = np.divide(z_pos, denominator_pos)
            fraction_neg = np.divide(z_neg, denominator_neg)

            fraction = (1 - beta) * fraction_pos + beta * fraction_neg

        else:
            try:
                denominator = np.matmul(output,
                                    weights)
            except ValueError:
                print("Dimension error in calc_r, outputs dimension {}, weights dimension {}".format(output.shape, weights.shape))
            

            if eps:
                denominator = denominator + eps * np.sign(denominator)
            # 0.0 im Nenner darf nicht sein, ersetze durch Kleine Zahl
            if 0.0 in denominator:
                #print("0 values in denominator, shape {} and values \n{}".format(denominator.shape, denominator))
                denominator[denominator==0.0]=0.00000001

            fraction = np.divide(nominator, denominator)
            if np.isnan(fraction).any():
                print("nan values in fraction, shape {} and values \n{}".format(fraction.shape, fraction))
                

        r_new = np.dot(fraction, r)

        return r_new


    # Funktion für Relevance Propagation NEU (beliebige anzahl schichten)
    def rel_prop(self, model: tf.keras.Sequential, input: np.ndarray, eps: float = 0.0, beta: float = None) -> np.ndarray:

        # Hilfsmodel zum Extrahieren der Outputs des Hidden Layers
        extractor = tf.keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

        features = extractor(np.array([input]))
        #extract number of features
        nFeatures = len(features)
        outputs = []
        weightMatrices = []
        for layer in range(nFeatures):
            outputs.append(features[layer].numpy())
            #print("appended output with shape {} to outputs ".format(features[layer].numpy().shape))
        # decrease nFeatures to collect weight matrices (one less than no. of features)
        nFeatures-=1
        weights = []
        for betweenLayer in range(nFeatures):
            weightMatrices.append(model.weights[betweenLayer].numpy())
            #print("appended weigth matrix with shape {} to outputs ".format(model.weights[betweenLayer].numpy().shape))
        # Equivalent to R^{(l+1)}, initiated with the output
        rel_prop_vector_2 = np.transpose(outputs[nFeatures])
        # again, decrease nFeatures by one to get access to the list indices -> Backwards calculation of input Relevance vector
        nFeatures-=1
        #print("relevance_propagation, nFeatures is {} before loop".format(nFeatures))
        while(nFeatures>=0):
            #rel_prop_vector_1 is equivalent to R^{(l)} from the notebook
            rel_prop_vector_1= self.calc_r(r=rel_prop_vector_2, 
                output=outputs[nFeatures],
                weights=weightMatrices[nFeatures],
                eps=eps,
                beta=beta)
            nFeatures-=1
            rel_prop_vector_2 = rel_prop_vector_1
            if np.isnan(rel_prop_vector_1).any():
                print("nan values in relevance_propagation vector shape {} and values \n{}".format(rel_prop_vector_1.shape, rel_prop_vector_1))
                return np.zeros(input.shape)

        # Finally, output of the relevance propagation is the same dimension as the flattened input vector, 
        # reshape it to original dimensions
        relevance = np.reshape(rel_prop_vector_1, input.shape)

        return relevance

    
    ###############################################################################################
    ######## MIN - MAX MODELL METHODEN ############################################################
    ###############################################################################################

    def get_higher_relevances(self, model: tf.keras.Sequential, input, eps : float = 0.0, beta : float = None):
         # Hilfsmodel zum Extrahieren der Outputs des Hidden Layers
        extractor = tf.keras.Model(inputs=model.inputs,
                                outputs=[layer.output for layer in model.layers])
        # Hilfsmodel berechnet Output der einzelnen Schichten für gegebenen Input
        features = extractor(np.array([input]))
        #print("feature extractor fuer input: ")
        #plot_mnist_image(input)
        # wg sum_pooling ist die Relevanz {R_l} = x_l
        high_relevances = features[3].numpy()
        
        # Outputs der einzelnen Schichten
        output_midlayer = features[2].numpy()

        #print("Dimensionen der Outputs: Mittlerer Layer {}, Hoeherer Layer {}".format(output_midlayer.shape, high_relevances.shape))
        #print("Werte der Outputs: Mittlerer Layer {}, Hoeherer Layer {}".format(output_midlayer, high_relevances))

        # Berechnung der Relevanzen des mittleren Layers
        mid_relevances = self.calc_r(r=np.transpose(high_relevances),
                    output=output_midlayer,
                    weights=model.weights[2].numpy(),
                    eps=eps,
                    beta=beta)
    
        return high_relevances, mid_relevances

#END class relevance_propagation



""" Klasse aus dem ersten Vortrag, wird diesmal mit dem SecIII model initialisiert"""
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
            model = getSection3Model()
            """AENDERUNG: BENUTZE HINGE LOSS UND SGD VERFAHREN"""
            model.compile(loss=tf.keras.losses.Hinge(),
                        optimizer=SGD(learning_rate = 0.0001),
                        metrics=['acc'])           
            

        if self.model_type == "load_dense":
            filepath = 'pretrained_models/sec_3_mnist_model.h5'
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

            filepath = 'pretrained_models/sec_3_mnist_model.h5'
            """AENDERUNG: SPEICHERE MODEL."""
            self.model.save(filepath)
        

    def predict(self, image):
        pred = int(self.model.predict(np.array([image]))[0][0])
        return pred
    
    #Nicht sinnvoll bei aktueller Modellarchitektur
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




""" NICHT BENUTZT
def prepareBinaryData(data, class_nb):
    train_images = data[0]
    train_labels =  data[1]
    test_images = data[2]
    test_labels = data[3]
    
    #make_binary_data
    train_labels = (train_labels==class_nb).astype(int)
    test_labels_new = (test_labels==class_nb).astype(int)
    
        # reduce train dataset
    one_indices = [i for i in range(train_labels.shape[0]) if train_labels[i]==1]
    zero_indices = [i for i in range(train_labels.shape[0]) if train_labels[i]==0]
    sampling = random.choices(zero_indices, k=3*len(one_indices))
    train_indices = one_indices + sampling
    print("Number of train indices: ", len(train_indices))
    train_images_new = np.asarray([train_images[i] for i in train_indices])
    print(train_images_new.shape)
    train_labels_new = np.asarray([train_labels[i] for i in train_indices])
    print("train_images_new shape {}, test_image shape {}, train_labels_new shape {}, test_labels shape {}".format(train_images_new.shape, test_images.shape, train_labels_new.shape, test_labels.shape))
    return train_images_new, train_labels_new, test_labels_new
    """