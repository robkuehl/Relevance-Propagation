from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.optimizers import Adam


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ntpath
from datetime import datetime
from PIL import Image
from pathlib import Path

import os
from os.path import join as pathjoin

from sklearn.model_selection import train_test_split

from src.data.get_voc_data import pascal_data_generator
from src.models.help_functions.models import get_cnn_model as get_model
#from models.custom_callback import CustomModelCheckpoint
from src.models.help_functions.custom_modelcheckpoint import ModelCheckpoint as CustomModelCheckpoint


class ml_cnn_classifier:
    
    """
    Klasse zum Trainieren eines Multi Label CNN (für den Pascal Voc Datensatz)
    - init params:
        :model_name: Festlegen welches CNN Modell verwendet werden soll. Mögliche Modelle befinden sich in src/models/help_functions/cnn_models.py
        :dataset: Datensatz für den das CNN trainiert werden soll (Auswahl aus den 3 Pascal Voc Varianten)
        :final_activation: Aktivierungsfunktion der letzten Schicht
        :loss: Loss function mit der das neuronale Netz trainiert wird
        :storage_path: Ort an dem das Modell mittels Model Checkpoint gespeichert wird
        :classes: Klassen aus dem Pascal Voc Datensatz auf denen das CNN trainiert werden soll. Falls keine angegeben, werden alle benutzt
        :model_path: Pfad der angegeben muss, falls ein bereits trainiertes Model verwendet werden soll. 
                     Modell werden i.d.R. unter Relevenace-Propagation/models/cnn abgelegt
        :monitor: Metrik die für Early Stopping verwendet werden soll. Falls None, kein Early Stopping
    - Methoden:
        :create_model: laden eines CNN mit den angegebenen Parametern und kompilieren
        :run_model: trainieren des CNN Modells
        :pred: Prediction mittels Modell für ein Bild aus dem Test Datensatz
        :eval: Evaluieren des Modells
    """
    
    def __init__(self, model_name:str, dataset:str, final_activation:str, loss:str, storage_path:Path, classes:list=None, model_path: str=None):
        self.model_path = model_path
        self.model_name = model_name
        self.dataset = dataset 
        self.final_activation = final_activation
        self.loss = loss
        self.metrics = [BinaryAccuracy(name='binary_accuracy'), Precision(name='precision'), Recall(name='recall')]
        self.monitor = 'val_precision'
        
        # Lade die Daten für das Training des CNN
        if 'pascal_voc' in dataset:
            pdg = pascal_data_generator()
            data, self.classes = pdg.get_training_data(classes, dataset)
        
        else:
            raise ValueError('Please enter an available dataset! Currently only Pascal Supported!')
            
        self.train_images = data['train_images']
        self.train_labels_df = data['train_labels_df']
        self.train_labels = data['train_labels_df'].values
        self.test_images = data['test_images']
        self.test_labels_df = data['test_labels_df']
        self.test_labels = self.test_labels_df.values
        self.input_shape = self.train_images[0].shape
        self.storage_path = storage_path
        self.output_shape = len(self.classes)

    

    def create_model(self):
        self.model = get_model(model_name=self.model_name, input_shape=self.input_shape, output_shape=self.output_shape, final_activation=self.final_activation)
        opt = Adam(learning_rate=0.0001)
        #opt = SGD(lr=0.001, momentum=0.9)
        self.model.compile(optimizer=opt,
                loss=self.loss,
                metrics=self.metrics)
        self.model.summary()
        
    
    """
    Methode zum Trainieren des Modells.
    Es wird mit Hilfe eines ImageDataGenerator sowie eines ModelCheckpoints und ggf. Early Stopping das gewählt CNN-Modell trainiert.
    - params:
        :batch_size (int): Sollte nicht zu groß gewählt werden. Unsere 6GB Grafikkarte schafft für Pascal Voc maximal 32
        :epochs (int):
    """
    
    def run_model(self, batch_size, epochs):
        # Falls im Konstruktor ein Modellpfad angegeben wurden, werden die Gewichte des Modells geladen und kein weiteres Training ist nötig.
        # Die History und alle weiteren während des Trainings erzeugten Ausgaben befinden sich im Ordner des Modells
        if self.model_path != None:
            self.model.load_weights(self.model_path)
            return 1
        else:
            pass
        
        # Splitte den Trainingsdatensatz in Training und Validierung damit wir Data Augmentation verwenden können
        self.train_images, self.validation_images, self.train_labels, self.validation_labels = train_test_split(self.train_images, self.train_labels, test_size=0.1)
        
        # Erstellen eines ImageDataGenerator für Data Augmentation
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                     rotation_range=50.0,
                                     width_shift_range = 0.1,
                                     height_shift_range = 0.1,
                                     #shear_range=0.1,
                                     zoom_range=0.1,
                                     horizontal_flip=True,
                                     vertical_flip = True
                                     )

        steps = int(self.train_images.shape[0] / batch_size)
        val_steps = int(self.validation_labels.shape[0]/batch_size)
        # Erstellen des Iterator
        it_train = datagen.flow(self.train_images, self.train_labels, batch_size=batch_size)
        
        # Erstellen der Iterator für Validierung und Evaluierung (keine Augmentation, nur gleiches Preprocessing)
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        it_val = test_datagen.flow(self.validation_images, self.validation_labels, batch_size=batch_size)
        it_eval = test_datagen.flow(self.test_images, self.test_labels, batch_size=batch_size)
        
        # Hier werden die Checkpoints gespeichert
        storage_path = pathjoin(self.storage_path, '{}_{}_{}.h5'.format(self.dataset, self.model_name, 'multilabel'))
        
        checkpoint = CustomModelCheckpoint(storage_path, verbose=1, save_best_only=True, save_weights_only=False, period=1, mode='max')
        if self.monitor != None:
            early = EarlyStopping(monitor=self.monitor, min_delta=0, patience=25, verbose=1, mode='max')
            self.history = self.model.fit(it_train, steps_per_epoch=steps, epochs=epochs, validation_data=it_val, validation_steps=val_steps, callbacks=[checkpoint, early], verbose=1)
        else:
            self.history = self.model.fit(it_train, steps_per_epoch=steps, epochs=epochs, validation_data=it_val, validation_steps=val_steps, callbacks=[checkpoint], verbose=1)
        
        # Evaluieren des Modells auf den Testdaten
        self.model.evaluate(it_eval, verbose=1, steps=int(self.test_images.shape[0]/batch_size))
        
        return 1
        

    """
    Methode üm für ein Bild des Trainingsdatensatzes eine Prediction zu machen. 
    - params:
        :i (int): Index des Bildes 
    """
    def pred(self, i):
        # Lades das Bild aus den test_images und predicte mittels Modell
        image = self.test_images[i]
        prediction = self.model.predict(np.asarray([preprocess_input(image)], dtype=np.float64))
        
        # Lade die Korrekten Label des Bildes und gib sie aus
        correct_labels = [self.classes[j] for j in range(self.test_labels[0].shape[0]) if self.test_labels[i][j]==1]
        print('Correct Label: {}\n'.format(correct_labels))
        for i in range(len(self.classes)):
            print('{}:\t\t {:.2f}'.format(self.classes[i], prediction[0][i]))
        print('\nNetwork detected:')
        for j in range(prediction[0].shape[0]):
            if prediction[0][j] >0.5:
                print('\t{} with {}%'.format(self.classes[j], prediction[0][j]*100))
        print("\n\n\n")
        return prediction
    

        
    # TODO: Beschreibung und kommentieren
    def eval(self):
        eval_df = pd.DataFrame(columns=['image', 'labels']+list(self.classes.values()))
        for i in range(self.test_images.shape[0]):
            image_name = self.test_labels_df.index[i]
            image = self.test_images[i]*1.0
            #model prediction
            prediction = self.model.predict(np.asarray([preprocess_input(image)], dtype=np.float64))[0]
            correct_labels = [self.classes[j] for j in range(self.test_labels[0].shape[0]) if self.test_labels[i][j]==1]
            image_result = [image_name, correct_labels]+[prediction[i] for i in range(prediction.shape[0])]
            eval_df = eval_df.append(pd.DataFrame([image_result], columns=eval_df.columns), ignore_index=True)
        
        return eval_df
