from tensorflow.keras.datasets import cifar10, mnist
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

def load_mnist():
    dirname = os.path.dirname(__file__)
    print(dirname)
    filename = os.path.join(dirname, '../../data/raw/mnist.pickle')
    
    if not os.path.isfile(Path(filename)):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        
        images = np.asarray(list(test_images)+list(train_images))
        labels = np.asarray(list(test_labels)+list(train_labels))
        
        mnist_data = {'images':images,
                    'labels':labels,
                    }
        with open(filename, 'wb') as f:
            pickle.dump(mnist_data, f)
    else:
        pass
    
    return filename
    

def get_mnist() -> dict:
    filename = load_mnist()
    with open(filename, 'rb') as f:
        mnist_data = pickle.load(f)
        
    return mnist_data
    

def load_cifar10():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../../data/raw/mnist.pickle')
    
    if not os.path.isfile(Path(filename)):
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        
        images = np.asarray(list(test_images)+list(train_images))
        labels = np.asarray(list(test_labels)+list(train_labels))
        
        cifar10_data = {'images':images,
                        'labels':labels,
                    }
        
        with open(filename, 'wb') as f:
            pickle.dump(cifar10_data, f)
    else:
        pass
    
    return filename
            
def get_cifar10() -> dict:
    filename = load_cifar10()
    with open(filename, 'rb') as f:
        cifar10_data = pickle.load(f)
        
    return cifar10_data
    
    
def get_raw_data(dataset: str) -> dict:
    if dataset == 'mnist':
        return get_mnist()
    elif dataset == 'cifar10':
        return get_cifar10()
    else:
        raise ValueError('Desired dataset is not available. Please choose mnist or cifar10')
    

def get_training_data(dataset: str, test_size: float) -> np.ndarray:
    data = get_raw_data(dataset)
    
    train_images, test_images, train_labels, test_labels = train_test_split(data['images'], data['labels'], test_size=test_size, random_state=42)
    
    return train_images, test_images, train_labels, test_labels