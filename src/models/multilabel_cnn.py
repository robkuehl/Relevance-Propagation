from data.get_data import get_fashion_mnist, get_cifar10
from data.get_voc_data import get_training_data
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
from keras.optimizers import SGD
import ntpath
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

voc_path='/home/robin/Documents/VOCdevkit/VOC2012'

class ml_cnn_classifier:
    
    def __init__(self, model_type:str, dataset:str, final_activation:str, loss:str, classes:list=None):
        self.model_type = model_type
        self.dataset = dataset 
        self.final_activation = final_activation
        self.loss = loss
        self.metric = 'binary_accuracy'
        self.monitor = 'val_binary_accuracy'
        if dataset == 'fashion_mnist':
            data, self.classes = get_fashion_mnist(encoded=True, training=True)
        if dataset == 'cifar10':
            data, self.classes = get_cifar10(encoded=True, training=True)
        elif 'voc' in dataset:
            data, self.classes = get_training_data(voc_path, classes, dataset)
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
        if self.model_type == 'model_1':
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
        
        if self.model_type == 'model_2':
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
            model.add(Dropout(rate=0.2))
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(rate=0.2))
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(rate=0.2))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(rate=0.2))
            model.add(Dense(self.output_shape, activation=self.final_activation))
        
        if self.model_type == 'model_3':
            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape))
            model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.2))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.2))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.2))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(self.output_shape, activation=self.final_activation))
            
            
        self.model = model
    

    def create_model(self):
        self.get_model()
        opt = 'adam'
        #opt = SGD(lr=0.001, momentum=0.9)
        self.model.compile(optimizer=opt,
                loss=self.loss,
                metrics=[self.metric])
        self.model.summary()
        
    
    def run_model(self, batch_size, epochs):
        # if GPU is used
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
             pass
         
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
        storage_path = pathjoin(self.storage_path, '{}_{}_{}_{}.h5'.format(self.dataset, self.model_type, 'multilabel', dt))
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
        print('\nNetwork detected:')
        for j in range(prediction[0].shape[0]):
            if prediction[0][j] >0.5:
                print('\t{} with {}%'.format(self.classes[j], prediction[0][j]*100))
        plt.show()
            
    
    def eval(self):
        scores = []
        predictions = []
        for i in range(self.test_images.shape[0]):
            image = self.test_images[i]
            prediction = self.model.predict(np.asarray([image]))
            predicted_labels =[j for j in range(prediction[0].shape[0]) if prediction[0][j]>0.5]
            correct_labels = [j for j in range(self.test_labels[0].shape[0]) if self.test_labels[i][j]==1]
            predictions.append((prediction[0], [(self.classes[j], np.around(prediction[0][j],2)) for j in predicted_labels], [self.classes[j] for j in correct_labels]))
            score = sum([1 for j in predicted_labels if j in correct_labels])/len(correct_labels)
            scores.append(score)
        
        return scores, predictions
                
                
            
        
            
            
        
            