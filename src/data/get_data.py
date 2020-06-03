from tensorflow.keras.datasets import cifar10, mnist
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

def get_mnist() -> dict:
    if not os.path.isfile(Path('../../data/raw/mnist.json')):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        
        images = np.asarray(list(test_images)+list(train_images))
        labels = np.asarray(list(test_labels)+list(train_labels))
        
        mnist_data = {'images':images,
                    'labels':labels,
                    }
        with open('../../data/raw/mnist.json', 'w') as f:
            json.dump(mnist_data, f)
            
    with open('../../data/raw/mnist.json') as f:
        mnist_data = json.load(f)
        
    return mnist_data

def get_cifar10() -> dict:
    if not os.path.isfile(Path('../../data/raw/cifar10.json')):
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        
        images = np.asarray(list(test_images)+list(train_images))
        labels = np.asarray(list(test_labels)+list(train_labels))
        
        cifar10_data = {'images':images,
                        'labels':labels,
                    }
        
        with open('../../data/raw/cifar10.json', 'w') as f:
            json.dump(cifar10_data, f)
            
    with open('../../data/raw/cifar10.json') as f:
        cifar10_data = json.load(f)
        
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