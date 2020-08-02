from models.binary_classifier import binary_classifier
from data import get_data


class one_vs_all_classifier:
    
    def __init__(self, dataset:str, model_type:str, test_size:float, loss_func:str):
        self.dataset = dataset
        self.model_type = model_type
        self.test_size = test_size
        self.load_data()
        self.create_binary_classifiers(loss_func)
        
    def load_data(self):
        if self.dataset=='mnist':
            self.data_dict = get_data.get_mnist()
        elif self.dataset == 'cifar10':
            self.data_dict = get_data.get_mnist()
        
    def create_binary_classifiers(self, loss_func):
        classes = set(list(self.data_dict['labels']))
        self.classifiers = [binary_classifier(model_type=self.model_type, dataset=self.dataset, class_nb=c) for c in classes]
        for cl in self.classifiers:
            cl.set_data(self.test_size)
            cl.set_model(loss_func=loss_func)
            
    def fit_classifiers(self, epochs, batch_size):
        for cl in self.classifiers:
            cl.fit_model(epochs, batch_size)
            
    def get_classifiers(self):
        return self.classifiers
        

    def predict(self, image):
        prediction = [(cl.predict(image), cl.class_nb) for cl in self.classifiers]
        return prediction