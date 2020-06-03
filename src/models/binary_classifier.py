import tensorflow as tf
from ..process import get_binary_data
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
 

class binary_classifier:
    
    def __init__(self, model_type: str, dataset: str, class_nb: int):
        self.model_type = model_type
        self.dataset = dataset
        self.class_nb = class_nb
        

    def set_data(self, test_size: float):
        self.train_images, self.test_images, self.train_labels, self.test_labels = get_binary_data.get_training_data(dataset=self.dataset, class_nb=self.class_nb, test_size=0.25)
        

    def set_model(self, loss_func):
        if loss_func == 'binary_crossentropy':
            last_activation = 'sigmoid'
        elif loss_func == 'hinge_loss':
            last_activation = 'tanh'
            
        if self.dataset == 'mnist':
            input_shape=(28,28)
        elif self.dataset == 'cifar10':
            input_shape=(32,32,3)
        
        if self.model_type == "dense":
            model = Sequential([
                Flatten(input_shape=input_shape),
                Dense(4096, activation='relu', use_bias=False),
                Dense(1, activation=last_activation, use_bias=False)
            ])

        model.summary()

        model.compile(loss=loss_func,
                    optimizer=Adam(),
                    metrics=['acc']
                    )

        self.model = model


    def fit_model(self, epochs: int, batch_size: int):
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