# TODO: Anpassen der Klasse nach Muster von multilabel_cnn. Akut nicht lauffähig!

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
import ntpath
from datetime import datetime
from PIL import Image

import os
from os.path import join as pathjoin

from sklearn.model_selection import train_test_split

from data.get_data import get_fashion_mnist, get_cifar10
from data.get_voc_data import pascal_data_generator
from models.cnn_models import get_model


class mc_cnn_classifier:
    
    def __init__(self, model_name:str, dataset:str, final_activation:str, loss:str, classes:list):
        self.model_name = model_name
        self.dataset = dataset 
        self.final_activation = 'softmax'
        self.loss = 'categorical_crossentropy'
        self.metric = 'accuracy'
        self.monitor = 'val_accuracy'
        if dataset == 'fashion_mnist':
            data, self.classes = get_fashion_mnist(encoded=True, training=True)
        if dataset == 'cifar10':
            data, self.classes = get_cifar10(encoded=True, training=True)
        elif 'pascal_voc' in dataset:
            pdg = pascal_data_generator()
            data, self.classes = pdg.get_training_data(classes, dataset)
        else:
            raise ValueError('Please enter an available dataset!')
        self.train_images = data['train_images']
        self.train_labels = data['train_labels']
        self.test_images = data['test_images']
        self.test_labels = data['test_labels']
        self.input_shape = self.train_images[0].shape
        self.storage_path = pathjoin(os.path.dirname(__file__), '../../models/cnn/')
        self.output_shape = len(self.classes)

    

    def create_model(self):
        self.model = get_model(model_name=self.model_name, input_shape=self.input_shape, output_shape=self.output_shape, final_activation=self.final_activation)
        opt = 'adam'
        #opt = SGD(lr=0.001, momentum=0.9)
        self.model.compile(optimizer=opt,
                loss=self.loss,
                metrics=[self.metric])
        self.model.summary()
        
    
    def run_model(self, batch_size, epochs):
        # data augmentation
        #create validation data
        self.train_images, self.validation_images, self.train_labels, self.validation_labels = train_test_split(self.train_images, self.train_labels, test_size=0.2)
        
        # create data generator
        datagen = ImageDataGenerator(rotation_range=50.0,
                                     width_shift_range = 0.1,
                                     height_shift_range = 0.1,
                                     shear_range=0.1,
                                     zoom_range=0.1,
                                     horizontal_flip=True,
                                     vertical_flip = True
                                     )
    
        steps = int(self.train_images.shape[0] / batch_size)
        # prepare iterator
        it_train = datagen.flow(self.train_images, self.train_labels, batch_size=batch_size)
                
        dt = datetime.now().strftime('%d_%m_%Y-%H')
        storage_path = pathjoin(self.storage_path, '{}_{}_{}_{}.h5'.format(self.dataset, self.model_name, 'multiclass', dt))
        
        checkpoint = ModelCheckpoint(storage_path, monitor=self.monitor, verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor=self.monitor, min_delta=0, patience=25, verbose=1, mode='auto')
        
        #self.model.fit(x=self.train_images, y=self.train_labels, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint,early], validation_split=0.2)
        history = self.model.fit_generator(it_train, steps_per_epoch=steps, epochs=epochs, validation_data=(self.validation_images, self.validation_labels), callbacks=[checkpoint,early], verbose=1)
        self.model.evaluate(x=self.test_images, y=self.test_labels, verbose=1, batch_size=batch_size)
        
        return ntpath.basename(storage_path), history
        

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
            