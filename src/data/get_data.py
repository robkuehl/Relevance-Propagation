from tensorflow.keras.datasets import cifar10, mnist
import json
import os
from pathlib import Path

def get_mnist() -> dict:
    if not os.path.isfile(Path('../../data/raw/mnist.json')):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        mnist_data = {'train_images':train_images,
                    'train_labels':train_labels,
                    'test_images':test_images,
                    'test_labels':test_labels
                    }
        with open('../../data/raw/mnist.json', 'w') as f:
            json.dump(mnist_data, f)
    
    
    with open('../../data/raw/mnist.json') as f:
        mnist_data = json.load(f)
        
    return mnist_data

def get_cifar10() -> dict:
    if not os.path.isfile(Path('../../data/raw/cifar10.json')):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        cifar10_data = {'train_images':train_images,
                    'train_labels':train_labels,
                    'test_images':test_images,
                    'test_labels':test_labels
                    }
        with open('../../data/raw/cifar10.json', 'w') as f:
            json.dump(cifar10_data, f)
            
    with open('../../data/raw/cifar10.json') as f:
        cifar10_data = json.load(f)
        
    return cifar10_data
    
def get_data(dataset: str) -> dict:
    if dataset == 'mnist':
        return get_mnist()
    elif dataset == 'cifar10':
        return get_cifar10()
    else:
        raise ValueError('Desired dataset is not available. Please choose mnist or cifar10')
    
