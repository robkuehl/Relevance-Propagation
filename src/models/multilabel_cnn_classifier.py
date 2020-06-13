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
from keras.optimizers import SGD
import ntpath

class cnn_classifier:
    
    def __init__(self, model_type:str, dataset:str, classification_type:str):
        self.model_type = model_type
        self.classification_type = classification_type
        self.dataset = dataset 
        if classification_type == 'multilabel':
            self.final_activation = 'sigmoid'
            self.loss = 'binary_crossentropy'
            self.metric = 'accuracy'
            self.monitor = 'val_accuracy'
        elif classification_type == 'multiclass':
            self.final_activation = 'softmax'
            self.loss = 'categorical_crossentropy'
            self.metric = 'accuracy'
            self.monitor = 'val_accuracy'
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
            model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=self.input_shape))
            model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.2))
            model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.2))
            model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.2))
            model.add(Flatten())
            model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
            model.add(Dropout(0.2))
            model.add(Dense(self.output_shape, activation=self.final_activation))
            
            
        self.model = model
    

    def create_model(self):
        self.get_model()
        #opt = 'adam'
        opt = SGD(lr=0.001, momentum=0.9)
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
        dt = datetime.now().strftime('%d_%m_%Y-%H')
        storage_path = pathjoin(self.storage_path, '{}_{}_{}_{}.h5'.format(self.dataset, self.model_type, self.classification_type, dt))
        checkpoint = ModelCheckpoint(storage_path, monitor=self.monitor, verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor=self.monitor, min_delta=0, patience=10, verbose=1, mode='auto')
        self.model.fit(x=self.train_images, y=self.train_labels, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint,early], validation_split=0.2)
        self.model.evaluate(x=self.test_images, y=self.test_labels, verbose=1)
        
        return ntpath.basename(storage_path)
        

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
            