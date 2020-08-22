"""
In dieser Datei stehen Methode zum laden der folenden Datensätze zur Verfügung:
    - MNIST
    - Fashion-MNIST
    - Cifar-10
"""

from tensorflow.keras.datasets import cifar10, mnist, fashion_mnist
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import random

dirname = os.path.dirname(__file__)


def get_fashion_mnist(encoded:bool, training:bool, test_size=0.2):
    """Laden des Fashion-Mnist Datensatzes. Der Datensatz wird lokal gespeichert und geladen für schnelles Wiederverwenden.

    Args:
        encoded (bool): Falls True werden die Label One-Hot-Encoded, s.d. die Label Einheitsvektoren und keine Strings sind
        training (bool): Falls True werden wird Train-test-split ausgeführt und ein dict mit entsprechenden Datensätzen zurückgegeben
        test_size (float, optional): Größe des Testsplits in traim-test-split. Defaults to 0.2.

    Returns:
        data [dict]: dictionary mit Daten nach ausgewählter Konfiguration durch encoded und training
        classes [list]: Klassen der zurückgegeben Daten
    """
    
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
    """Laden des Mnist Datensatzes. Der Datensatz wird lokal gespeichert und geladen für schnelles Wiederverwenden.

    Returns:
        data [dict]: dictionary mit Daten nach ausgewählter Konfiguration durch encoded und training
        classes [list]: Klassen der zurückgegeben Daten
    """
    filename = os.path.join(dirname, '../../data/raw/mnist.pickle')
    if not os.path.isfile(Path(filename)):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        images = np.asarray(list(test_images)+list(train_images))
        labels = np.asarray(list(test_labels)+list(train_labels))
        
        data = {
                'images':images,
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
    """Laden des Cifar-10 Datensatzes. Der Datensatz wird lokal gespeichert und geladen für schnelles Wiederverwenden.

    Args:
        encoded (bool): Falls True werden die Label One-Hot-Encoded, s.d. die Label Einheitsvektoren und keine Strings sind
        training (bool): Falls True werden wird Train-test-split ausgeführt und ein dict mit entsprechenden Datensätzen zurückgegeben
        test_size (float, optional): Größe des Testsplits in traim-test-split. Defaults to 0.2.

    Returns:
        data [dict]: dictionary mit Daten nach ausgewählter Konfiguration durch encoded und training
        classes [list]: Klassen der zurückgegeben Daten
    """
    
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

    


    
def get_mnist_binary(class_nb:int, test_size):
    """Erstellt einen Datensatz auf Basis des MNIST-Datensatzes der nur für die gewählte Klasse class_nb das Label 1 enthält.
        Benötigt werden diese Daten für das Min-Max-Modell.
        Um zu garantieren dass der Classifier die gewählte Klasse unterscheiden kann, werden zufällige Sample der anderen Klassen dem Datensatz hinzugefügt.
        Abschließend werden die Werte der Pixel auf (-0.5, 1.5) skaliert.
    
    Args:
        class_nb (int): Klasse die der buinäre Classifier erkennen soll
        test_size ([type]): Größe des Test-Splits in train_test_split

    Returns:
        train_images, test_images, train_labels, test_labels (numpy.ndarray)
    """
    sample_size = 3
    
    data, _ = get_mnist()
    
    images = data['images']
    labels = data['labels']
    labels = (labels==class_nb).astype(int)
        
    # reduce train dataset
    one_indices = [i for i in range(labels.shape[0]) if labels[i]==1]
    zero_indices = [i for i in range(labels.shape[0]) if labels[i]==0]
    sampling = random.choices(zero_indices, k=sample_size*len(one_indices))
    indices = one_indices + sampling
    #print("Number of train indices: ", len(indices))
        
    images = np.asarray([images[i] for i in indices])
    labels = np.asarray([[labels[i]] for i in indices])

    images = images.reshape((27300, 784,))
    scaler = MinMaxScaler(feature_range=(-0.5, 1.5))
    images = scaler.fit_transform(X=images).reshape((27300, 28, 28))

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size, random_state=42)
    print("Train images: {}, train labels: {}".format(train_images.shape[0], train_labels.shape[0]))     
        
    return train_images, test_images, train_labels, test_labels




def get_fashion_mnist_binary(class_nb:int, test_size):
    """Erstellt einen Datensatz auf Basis des MNIST-Datensatzes der nur für die gewählte Klasse class_nb das Label 1 enthält.
        Benötigt werden diese Daten für das Min-Max-Modell.
        Um zu garantieren dass der Classifier die gewählte Klasse unterscheiden kann, werden zufällige Sample der anderen Klassen dem Datensatz hinzugefügt.
        Abschließend werden die Werte der Pixel auf (-0.5, 1.5) skaliert.
    
    Args:
        class_nb (int): Klasse die der buinäre Classifier erkennen soll
        test_size ([type]): Größe des Test-Splits in train_test_split

    Returns:
        train_images, test_images, train_labels, test_labels (numpy.ndarray)
    """
    sample_size = 3
    
    data, classes = get_fashion_mnist(encoded=False, training=False)
    
    true_label = classes[class_nb]
    
    images = data['images']
    labels = data['labels']
    labels = (labels==true_label).astype(int)
        
    # reduce train dataset
    one_indices = [i for i in range(labels.shape[0]) if labels[i]==1]
    zero_indices = [i for i in range(labels.shape[0]) if labels[i]==0]
    sampling = random.choices(zero_indices, k=sample_size*len(one_indices))
    indices = one_indices + sampling
    #print("Number of train indices: ", len(indices))
        
    images = np.asarray([images[i] for i in indices])
    labels = np.asarray([[labels[i]] for i in indices])

    images = images.reshape((27300, 784,))
    scaler = MinMaxScaler(feature_range=(-0.5, 1.5))
    images = scaler.fit_transform(X=images).reshape((27300, 28, 28))

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size, random_state=42)
    print("Train images: {}, train labels: {}".format(train_images.shape[0], train_labels.shape[0]))     
        
    return train_images, test_images, train_labels, test_labels




