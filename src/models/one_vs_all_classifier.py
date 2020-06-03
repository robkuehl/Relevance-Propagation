from .binary_classifier import binary_classifier
from ..data import get_data


class one_vs_all_classifier:
    
    def __init__(self, dataset, model_type):
        self.dataset = dataset
        self.model_type = model_type
        self.load_data()
        self.create_binary_classifiers()
        
    def load_data(self):
        if self.dataset=='mnist':
            self.data_dict = get_data.get_mnist()
        elif self.dataset == 'cifar10':
            self.data_dict = get_data.get_mnist()
        
    def create_binary_classifiers(self):
        classes = set(list(self.data_dict['labels']))
        self.classifiers = [binary_classifier(model_type=self.model_type, dataset=self.dataset, class_nb=c) for c in classes]
        for cl in self.classifiers:
            cl.set_data()
            cl.set_model()
            
    def fit_classifiers(self, epochs, batch_size):
        for cl in self.classifiers:
            cl.fit_model(epochs, batch_size)
            
    def get_classifiers(self):
        return self.classifiers
        

    def predict(self, image):
        prediction = [(cl.predict(image), cl.class_nb) for cl in self.classifiers]
        return prediction