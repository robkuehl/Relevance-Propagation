from tensorflow.keras.datasets import cifar10, mnist, fashion_mnist
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder

dirname = os.path.dirname(__file__)


    
def get_fashion_mnist(encoded:bool, training:bool, test_size=0.2):
    filename = os.path.join(dirname, '../../data/raw/fashion_mnist.pickle')
    if not os.path.isfile(Path(filename)):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        images = np.asarray(list(test_images)+list(train_images))
        labels = np.asarray(list(test_labels)+list(train_labels))
        
        data = {'images':images,
                    'labels':labels,
                    }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    else:
        pass
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        images = data['images']
        labels = data['labels']
        
        
    classes = {
                    0 : 'T-Shirt',
                    1 : 'Trouser',
                    2 : 'Pullover',
                    3 : 'Dress',
                    4 : 'Coat',
                    5 : 'Sandal',
                    6 : 'Shirt',
                    7 : 'Sneaker',
                    8 : 'Bag',
                    9 : 'Ankle boot'
                }
    
    if encoded == True:
        labels=labels.reshape(-1,1)
        onehot_encoder = OneHotEncoder(sparse=False)
        labels = onehot_encoder.fit_transform(labels)
        data = {'images':images,
                    'labels':labels,
                    }
        
    if training == True:
        dim1, dim2, dim3 = images.shape
        train_images, test_images, train_labels, test_labels = train_test_split(images.reshape(dim1,dim2,dim3,1), labels, test_size=test_size, random_state=42)
        data = {'train_images': train_images,
                'train_labels': train_labels,
                'test_images': test_images,
                'test_labels': test_labels
        }
    
    return data, classes

    
def get_mnist():
    filename = os.path.join(dirname, '../../data/raw/mnist.pickle')
    if not os.path.isfile(Path(filename)):
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
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    classes = [i for i in range(10)]
    
    return data, classes

    
def get_cifar10(encoded:bool, training:bool, test_size=0.2):
    filename = os.path.join(dirname, '../../data/raw/cifar10.pickle')
    if not os.path.isfile(Path(filename)):
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        images = np.asarray(list(test_images)+list(train_images))/255.0
        labels = np.asarray(list(test_labels)+list(train_labels))
        
        data = {'images':images,
                    'labels':labels,
                    }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    else:
        pass
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        images = data['images']
        labels = data['labels']
    
    
    classes = { 0 : 'airplane',
                    1 : 'automobile',
                    2 : 'bird',
                    3 : 'cat',
                    4 : 'deer',
                    5 : 'dog',
                    6 : 'frog',
                    7 : 'horse',
                    8 : 'ship',
                    9 : 'truck'
                }
    
    if encoded == True:
        onehot_encoder = OneHotEncoder(sparse=False)
        labels = onehot_encoder.fit_transform(labels)
        data = {'images':images,
                    'labels':labels,
                    }
        
    if training == True:
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size, random_state=42)
        data = {'train_images': train_images,
                'train_labels': train_labels,
                'test_images': test_images,
                'test_labels': test_labels
        }
    
    return data, classes

    


    
    

    
