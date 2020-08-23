import tensorflow as tf
from src.data.get_data import get_mnist_binary
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from os.path import join as pathjoin
from datetime import datetime

dirname = filepath = os.path.dirname(__file__)
storage_path = pathjoin(dirname, '..','..','models', 'minmax')

# Der Monatvon Classifier implementiert das Model aus Kapitel IV "Application to Deep Networks", Abschnitt C aus Montavon et al 
# welches zur Erkennung von MNIST Bildern genutzt wird
# Das Modell hat die folgende Struktur
#   Flattened Input
#   Dense Layer mit 400 Detektionsneuronen, relu aktivierung
#   Pooling Operation: 400 neuronen jeweils 4 summieren -> 100 neuronen Output
#   Dense Layer mit 400 Detektionsneuronen, relu aktivierung
#   Globales sum-pooling zu einem outputneuron. Output soll ungefähr 1 sein, falls Zahl erkannt wurde, 0 sonst
# Das Modell kann nur binäre Entscheidungen treffen und wird daher immer für eine Klasse von Bildern trainiert
# 


class Montavon_Classifier:
    """
    :param class_nb: Definiert für welche Klasse (0-9) das Modell trainiert werden soll
    :param load_model: Setzt fest ob ein gespeichertes bereits trainiertes Modell geladen werden soll. Falls nicht, wird es neu trainiert
    """
    
    def __init__(self, class_nb: int, load_model:bool):
        self.class_nb = class_nb
        self.load_model = load_model
        self.storage_path = pathjoin(storage_path, "montavon_classifier_{}".format(self.class_nb))
        if not os.path.isdir(self.storage_path):
            os.makedirs(self.storage_path)
        

    def set_data(self, test_size: float):
        """
        Args:
            test_size (float): defines the size of the test data split 
        """
        self.train_images, self.test_images, self.train_labels, self.test_labels = get_mnist_binary(class_nb=self.class_nb, test_size=test_size)

        self.classes = [0,1]
        

 
    def set_model(self):
        """
        We create the model which is described at the top of the file
        """
        if (self.load_model and os.path.isdir(self.storage_path)):
            print('Load model')
            self.model = tf.keras.models.load_model(pathjoin(self.storage_path, 'model.h5'))
            return
        elif(self.load_model and not os.path.isdir(self.storage_path)):
            print("No model file to load. Model will be fitted!")
        #Eingebaute Funktion, die die Uebergangsmatrix mit 1en initialisiert.
        ones_initializer = tf.keras.initializers.Ones()
        model = Sequential()
        model.add(Flatten(input_shape=self.train_images[0].shape))
        model.add(Dense(400, activation='relu', use_bias = False))
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
        model.layers[2].set_weights([np.transpose(self.getSumPoolingWeights(400,100))])
        
        self.model = model
        self.model.compile(loss='binary_crossentropy',
                        optimizer=Adam(learning_rate = 0.0001),
                        metrics=['acc'])
       
        self.model.summary()

    
    
    def getSumPoolingWeights(self, inputDim, outputDim):
        """
        Definiert eine Matrix, die inputDim mittels Sum Pooling auf outputDim reduziert
        Wird ausschließlich in set_model verwendet 
        """
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
        """Fit the montavon model with binary MNIST data for the selected class

        Args:
            epochs (int): number of epochs to train
            batch_size (int): size of the batches during training
        """
        if(self.load_model and os.path.isdir(self.storage_path)):
            print("Model has been load, no need to train!")
            return
        early = EarlyStopping(monitor='val_acc', patience=25, verbose=2)
        checkpoint = ModelCheckpoint(monitor='val_acc', filepath=pathjoin(self.storage_path, 'model.h5'), verbose=2, safe_best_only=True)
        self.model.fit(
            self.train_images,
            self.train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=2,
            callbacks=[checkpoint, early]
        )

    def predict_train_image(self, index:int):
        """Make a prediction for an image from the training set. Later on used to train the relevance model
        Args:
            index (int): index of the picture in the training data set
        Returns:
            [int]: 1 or 0 
        """
        image = self.train_images[index]
        pred = int(self.model.predict(np.array([image]))[0][0])
        return pred
    
    
    def predict_test_image(self, index):
        """Make a prediction for an image from the test set.
        Args:
            index (int): index of the picture in the test data set
        Returns:
            [int]: 1 or 0 
        """
        image = self.test_images[index]
        pred = int(self.model.predict(np.array([image]))[0][0])
        return pred
    
    
    def non_trivial_accuracy(self):
        """Calculates accuracy for test images with label = 1 (which make up only 1/4 of the training data set)
        Returns:
            [float]: percentage of the correct classified images
        """
        answers = []
        for i in range(len(list(self.test_labels))):
            if self.test_labels[i]==1:
                answers.append(int(self.model.predict(np.array([self.test_images[i]]))[0][0]))
                
        return sum(answers)/len(answers)
    
    
    def evaluate(self, batch_size:int):
        """Evaluate the model on the test data set
        Args:
            batch_size (int): size if batches in forward passes
        Returns:
            [float]: acuuracy of the prediction
        """
        _ , acc = self.model.evaluate(self.test_images, self.test_labels,
                                batch_size=batch_size)
        return acc