# TODO: Anpassen der Klasse nach Muster von multilabel_cnn. Akut nicht lauffähig!

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import ntpath
from datetime import datetime
from PIL import Image

import os
from os.path import join as pathjoin
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data.get_data import get_fashion_mnist, get_cifar10, get_mnist
from src.data.get_voc_data import pascal_data_generator
from src.models.help_functions.models import get_cnn_model, get_dense_model


class mc_classifier:
    
    def __init__(self, model_type:str, model_name:str, dataset:str, storage_path:Path, model_path: str=None):
        self.model_path = model_path
        self.model_name = model_name
        self.model_type = model_type
        self.dataset = dataset 
        self.final_activation = 'softmax'
        self.loss = 'categorical_crossentropy'
        self.metric = 'accuracy'
        self.monitor = 'val_accuracy'
        
        if dataset == 'mnist':
            data, self.classes = get_mnist()
        if dataset == 'fashion_mnist':
            data, self.classes = get_fashion_mnist(encoded=True, training=True)
        if dataset == 'cifar10':
            data, self.classes = get_cifar10(encoded=True, training=True)
        
        else:
            raise ValueError('Please enter an available dataset!')
        
        self.train_images = data['train_images']
        self.train_labels = data['train_labels']
        self.test_images = data['test_images']
        self.test_labels = data['test_labels']
        self.input_shape = self.train_images[0].shape
        self.output_shape = len(self.classes)
        self.storage_path = storage_path
    

    def create_model(self):
        if self.model_type == 'cnn':
            self.model = get_cnn_model(model_name=self.model_name, input_shape=self.input_shape, output_shape=self.output_shape, final_activation=self.final_activation)
        elif self.model_type == 'dense':
            self.model = get_dense_model(model_name=self.model_name, input_shape=self.input_shape, output_shape=self.output_shape, final_activation=self.final_activation)
        else:
            raise ValueError("model_type should be dense or cnn")
        opt = Adam(learning_rate=0.0001)
        #opt = SGD(lr=0.001, momentum=0.9)
        self.model.compile(optimizer=opt,
                loss=self.loss,
                metrics=[self.metric])
        self.model.summary()
        
    
    def run_model(self, batch_size, epochs):
        if self.model_path != None:
            self.model.load_weights(self.model_path)
            return 1
        else:
            pass
        # data augmentation
        #create validation data
        self.train_images, self.validation_images, self.train_labels, self.validation_labels = train_test_split(self.train_images, self.train_labels, test_size=0.2)
        
        '''
        datagen = ImageDataGenerator(rotation_range=50.0,
                                     width_shift_range = 0.1,
                                     height_shift_range = 0.1,
                                     horizontal_flip=True,
                                     vertical_flip = True
                                     )
        '''
        datagen = ImageDataGenerator()
    
        steps = int(self.train_images.shape[0] / batch_size)
        val_steps = int(self.validation_labels.shape[0]/batch_size)
        # Erstellen des Iterator
        it_train = datagen.flow(self.train_images, self.train_labels, batch_size=batch_size)
        
        # Erstellen der Iterator für Validierung und Evaluierung (keine Augmentation, nur gleiches Preprocessing)
        test_datagen = ImageDataGenerator()
        it_val = test_datagen.flow(self.validation_images, self.validation_labels, batch_size=batch_size)
        it_eval = test_datagen.flow(self.test_images, self.test_labels, batch_size=batch_size)
        
        # Hier werden die Checkpoints gespeichert
        storage_path = pathjoin(self.storage_path, '{}_{}_{}.h5'.format(self.dataset, self.model_name, 'multiclass'))

        checkpoint = ModelCheckpoint(storage_path, monitor=self.monitor, verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor=self.monitor, min_delta=0, patience=25, verbose=1, mode='auto')
        
        self.history = self.model.fit(it_train, steps_per_epoch=steps, epochs=epochs, validation_data=it_val, validation_steps=val_steps, callbacks=[checkpoint, early], verbose=1)
        self.model.evaluate(it_eval, verbose=1, steps=int(self.test_images.shape[0]/batch_size))
        
        return 1
        

    def pred(self, i):
        image = self.test_images[i]
        prediction = self.model.predict(np.array([image]))
        plt.imshow(Image.fromarray(np.uint8(image*255)))
        print('Correct Label: {}\n'.format(self.classes[np.argmax(self.test_labels[i])]))
        for i in range(len(self.classes)):
            print('{}:\t\t {:.2f}'.format(self.classes[i], prediction[0][i]))
        print('\nNetwork Decision: {}'.format(self.classes[np.argmax(prediction)]))
        plt.show()
            
    
    def eval(self):
        in_top1 = []
        in_top3 = []
        top_3_predictions = []
        for i in range(self.test_images.shape[0]):
            image = self.test_images[i]
            prediction = self.model.predict(np.asarray([image]))
            top_3 = [self.classes[j] for j in list(prediction[0].argsort()[-3:][::-1])]
            in_top1.append(int(np.argmax(self.test_labels[i])==np.argmax(prediction[0])))
            is_in_top3 = int(np.argmax(self.test_labels[i]) in list(prediction[0].argsort()[-3:][::-1]))
            in_top3.append(is_in_top3)
            top_3_predictions.append((is_in_top3, top_3, self.classes[np.argmax(self.test_labels[i])]))
        top1_score = sum(in_top1)/len(in_top1)
        top3_score = sum(in_top3)/len(in_top3)
        
        return top1_score, top3_score, top_3_predictions
            