import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../data')

from ..data import get_data
import os
from pathlib import Path
import random
import json
import numpy as np
from sklearn.model_selection import train_test_split

def get_raw_data(dataset: str, class_nb: int) -> dict:
        filename = '{}_binary_data_class_{}.json'.format(dataset, class_nb)
        sample_size = 3
        
        if not os.path.isfile(Path('../../data/processed/{}'.format(filename))):
            data_dict = get_data.get_raw_data(dataset)
            images = data_dict['images']
            labels = data_dict['labels']
            
            labels = (labels==class_nb).astype(int)
            
            # reduce train dataset
            one_indices = [i for i in range(labels.shape[0]) if labels[i]==1]
            zero_indices = [i for i in range(labels.shape[0]) if labels[i]==0]
            sampling = random.choices(zero_indices, k=sample_size*len(one_indices))
            indices = one_indices + sampling
            #print("Number of train indices: ", len(indices))
            
            images = np.asarray([images[i] for i in indices])
            labels = np.asarray([labels[i] for i in indices])
            
            data_dict = {'images': images,
                         'labels': labels}
            
            with open(Path('../../data/processed/{}'.format(filename)), 'w') as f:
                json.dump(data_dict, f)
            
        with open(Path('../../data/processed/{}'.format(filename))) as f:
            data_dict = json.load(f)
        
        return data_dict
    

def get_training_data(dataset: str, class_nb: int, test_size: float) -> np.ndarray:
    data = get_raw_data(dataset, class_nb)
    
    train_images, test_images, train_labels, test_labels = train_test_split(data['images'], data['labels'], test_size=test_size, random_state=42)
    
    return train_images, test_images, train_labels, test_labels
        