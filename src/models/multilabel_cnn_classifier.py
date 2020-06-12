from data.get_data import get_training_data, get_classes
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
import matplotlib.pyplot as plt
from os.path import join as pathjoin
import os


# if GPU is used
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

class cnn_classifier:
    
    def __init__(self, model_type:str, dataset:str):
        self.model_type = model_type
        self.dataset = dataset 
        self.train_images, self.test_images, self.train_labels, self.test_labels = get_training_data(dataset=dataset, test_size=0.2, encoded=True)
        self.input_shape = self.train_images[0].shape
        self.storage_path = pathjoin(os.path.dirname(__file__), '../../models/cnn/')
        self.classes = get_classes(dataset)
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
        
        
        self.model = model
    

    def create_model(self):
        self.get_model()
        self.model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
        self.model.summary()
        
    
    def run_model(self, batch_size, epochs):
        dt = datetime.now().strftime('%d/%m/%Y-%H')
        storage_path = pathjoin(self.storage_path, '{}_{}_{}.h5'.format(self.dataset, self.model_type, dt))
        checkpoint = ModelCheckpoint(storage_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
        self.model.fit(x=self.train_images, y=self.train_labels, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint,early], validation_split=0.1)
        self.model.evaluate(x=self.test_images, y=self.test_labels)
        

    def pred(self, i):
        prediction = self.model.predict(np.array([self.test_images[i]]))
        plt.imshow(self.test_images[i])
        print('Correct Label: {}\n'.format(self.classes[self.test_labels[i][0]]))
        for i in range(len(self.classes)):
            print('{}:\t\t {:.2f}'.format(self.classes[i], prediction[0][i]))
        print('\nNetwork Decision: {}'.format(self.classes[np.argmax(prediction)]))
            