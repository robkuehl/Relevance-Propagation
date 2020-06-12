from data.get_data import get_fashion_mnist, get_cifar10
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, BatchNormalization, Dropout
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
import matplotlib.pyplot as plt
from os.path import join as pathjoin
import os
from PIL import Image

class cnn_classifier:
    
    def __init__(self, model_type:str, dataset:str):
        self.model_type = model_type
        self.dataset = dataset 
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
        self.storage_path = pathjoin(os.path.dirname(__file__), '../../models/cnn/')
        self.output_shape = len(self.classes)


    def get_model(self):
        if self.model_type == 'simple_model':
            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(Conv2D(256, kernel_size=3, activation='relu', ))
            model.add(Flatten())
            model.add(Dense(40, activation='relu'))
            model.add(Dense(self.output_shape, activation='sigmoid'))
        
        if self.model_type == 'complex_model':
            model = Sequential()
            model.add(Conv2D(64, (3,3), activation='relu', input_shape=self.input_shape))
            model.add(MaxPooling2D((2,2), padding='same'))
            model.add(BatchNormalization())
            model.add(Conv2D(128, (3,3), activation='relu', input_shape=self.input_shape))
            model.add(MaxPooling2D((2,2), padding='same'))
            model.add(BatchNormalization())
            model.add(Conv2D(256, (3,3), activation='relu', input_shape=self.input_shape))
            model.add(MaxPooling2D((2,2), padding='same'))
            model.add(BatchNormalization())
            model.add(Conv2D(512, (3,3), activation='relu', input_shape=self.input_shape))
            model.add(MaxPooling2D((2,2), padding='same'))
            model.add(BatchNormalization())
            model.add(Flatten())
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(rate=0.1))
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(rate=0.1))
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(rate=0.1))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(rate=0.1))
            model.add(Dense(self.output_shape, activation='sigmoid'))
            
        self.model = model
    

    def create_model(self):
        self.get_model()
        self.model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
        self.model.summary()
        
    
    def run_model(self, batch_size, epochs):
        # if GPU is used
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
             pass
        dt = datetime.now().strftime('%d_%m_%Y-%H')
        storage_path = pathjoin(self.storage_path, '{}_{}_{}.h5'.format(self.dataset, self.model_type, dt))
        checkpoint = ModelCheckpoint(storage_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto')
        self.model.fit(x=self.train_images, y=self.train_labels, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint,early], validation_split=0.1)
        self.model.evaluate(x=self.test_images, y=self.test_labels, verbose=1)
        

    def pred(self, i):
        image = self.test_images[i]
        prediction = self.model.predict(np.array([image]))
        plt.imshow(Image.fromarray(image))
        print('Correct Label: {}\n'.format(self.classes[np.argmax(self.test_labels[i])]))
        for i in range(len(self.classes)):
            print('{}:\t\t {:.2f}'.format(self.classes[i], prediction[0][i]))
        print('\nNetwork Decision: {}'.format(self.classes[np.argmax(prediction)]))
            
    
    def eval(self):
        predicted_correct = []
        for i in range(self.test_images.shape[0]):
            image = self.test_images[i]
            prediction = self.model.predict(np.array([image]))
            decision = np.argmax(prediction)
            label = np.argmax(self.test_labels[i])
            predicted_correct.append(int(label==decision))
        return sum(predicted_correct)/len(predicted_correct)
            