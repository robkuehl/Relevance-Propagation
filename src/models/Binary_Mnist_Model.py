import tensorflow as tf
from src.data.get_data import get_mnist_binary
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.optimizers import SGD
 

class Montavon_Classifier:
    
    def __init__(self, class_nb: int):
        self.class_nb = class_nb
        

    def set_data(self, test_size: float):
        self.train_images, self.test_images, self.train_labels, self.test_labels = get_mnist_binary(class_nb=self.class_nb, test_size=test_size)
        self.classes = set(list(self.train_labels))
        

    """
    Das Modell aus Sec III in Montavon et al
    Besteht aus: 
    Flattened Input
    Dense Layer mit 400 Detektionsneuronen, relu aktivierung
    Pooling Operation: 400 neuronen jeweils 4 summieren -> 100 neuronen Output
    Dense Layer mit 400 Detektionsneuronen, relu aktivierung
    Globales sum-pooling zu einem outputneuron. Output soll ungefähr 1 sein, falls Zahl erkannt wurde, 0 sonst
    """ 
    def set_model(self):
        #Eingebaute Funktion, die die Uebergangsmatrix mit 1en initialisiert.
        ones_initializer = tf.keras.initializers.Ones()
        model = Sequential()
        model.add(Dense(400, activation='relu', use_bias = False, input_shape=self.train_images[0].shape))
        #Kernel initializer sorgt dafuer, dass die Gewichtsmatrix die geforderte Pooling Operation realisiert
        custom_pooling = Dense(100, activation = 'relu', use_bias = False, kernel_initializer = ones_initializer)
        #Gewichte sollen nicht veraendert werden
        custom_pooling.trainable=False
        model.add(custom_pooling)
        model.add(Dense(400, activation='relu', use_bias = False))
        #Gleiches wie oben, kernel wird mit 1en initialisiert und nicht trainierbar -> sum-pooling
        sum_pooling = Dense(1, activation = 'sigmoid', use_bias = False, kernel_initializer = ones_initializer)
        sum_pooling.trainable = False
        model.add(sum_pooling)
        #print("list of weights [0] shape: {}, [1] shape {}".format(list_of_weights[0].shape, list_of_weights[1].shape))
        model.layers[1].set_weights([np.transpose(self.getSumPoolingWeights(400,100))])
        
        self.model = model
        self.model.compile(loss='binary_crossentropy',
                        optimizer=SGD(learning_rate = 0.0001),
                        metrics=['acc'])
       
        model.summary()

    
    """
    Definiert eine Matrix, die inputDim mittels Sum Pooling auf outputDim reduziert
    Wird ausschließlich in set_model verwendet 
    """
    def getSumPoolingWeights(self, inputDim, outputDim):
        #Bestimme die Anzahl an Neuronen, die auf ein Outputneuron summiert werden
        pool_ratio = int(inputDim /outputDim)
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



    def fit_model(self, epochs: int, batch_size: int):
        self.model.fit(
            self.train_images,
            self.train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=2
        )

    def predict_train_image(self, index):
        image = self.train_images[index]
        pred = int(self.model.predict(np.array([image]))[0][0])
        return pred
    
    
    def predict_test_image(self, index):
        image = self.test_images[index]
        pred = int(self.model.predict(np.array([image]))[0][0])
        return pred
    
    
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