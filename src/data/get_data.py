from tensorflow.keras.datasets import cifar10, mnist, fashion_mnist
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

def load_keras_data(dataset: str):
    dirname = os.path.dirname(__file__)
    print(dirname)
    filename = os.path.join(dirname, '../../data/raw/{}.pickle'.format(dataset))
    
    if not os.path.isfile(Path(filename)):
        if dataset == "mnist":
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        elif dataset == "cifar10"
            (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        elif dataset == "fashion_mnist"
            (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        
        images = np.asarray(list(test_images)+list(train_images))
        labels = np.asarray(list(test_labels)+list(train_labels))
        
        data = {'images':images,
                    'labels':labels,
                    }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    else:
        pass
    
    return filename


def get_keras_data(dataset) -> dict:
    filename = load_keras_data(dataset)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        
    return data


    
    
def get_raw_data(dataset: str) -> dict:
    if dataset == 'mnist' or dataset == 'cifar10' or dataset == 'fasion_mnist':
        return get_keras_data(dataset)
    else:
        raise ValueError('Desired dataset is not available. Please choose mnist or cifar10')
    

def get_training_data(dataset: str, test_size: float) -> np.ndarray:
    data = get_raw_data(dataset)
    
    train_images, test_images, train_labels, test_labels = train_test_split(data['images'], data['labels'], test_size=test_size, random_state=42)
    
    return train_images, test_images, train_labels, test_labels