import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
 

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
        print("test_images shape {}".format(test_images.shape))
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
        
        
        

    def set_model(self):
        
        if self.dataset == 'mnist':
            input_shape=(28,28)
        elif self.dataset == 'cifar10':
            input_shape=(32,32,3)
        
        if self.model_type == "dense":
            model = Sequential([
                Flatten(input_shape=input_shape),
                Dense(4096, activation='relu', use_bias=False),
                Dense(1, activation='sigmoid', use_bias=False)
            ])

        model.summary()

        model.compile(loss='binary_crossentropy',
                    optimizer=Adam(),
                    metrics=['acc'])

        self.model = model


    def fit_model(self, epochs: int, batch_size: int):
        if tf.config.list_physical_devices('GPU'):
            dev='/gpu:0'
        else:
            dev = '/cpu:0'
        with tf.device(dev):
            self.model.fit(
                self.train_images,
                self.train_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(self.test_images, self.test_labels),
                verbose=2
            )

    def predict(self, image):
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