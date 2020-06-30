#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 09:42:09 2020

@author: robin
"""
#import time
import os
import pickle
#import random
import matplotlib.pyplot as plt

import tensorflow as tf
#import numpy as np
#import matplotlib.pyplot as plt
#from tensorflow_addons.metrics import HammingLoss
from datetime import datetime
from pathlib import Path

from models.multilabel_cnn import ml_cnn_classifier
#from models.multiclass_cnn import mc_cnn_classifier
#from rel_prop.rel_prop import rel_prop
#from rel_prop.help_func import MidpointNormalize
from contextlib import redirect_stdout


os.environ['KMP_DUPLICATE_LIB_OK']='True'

# if GPU is used
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass



dirname = os.path.dirname(__file__)
mc_dictpath = os.path.join(dirname, '..', 'models','cnn', 'multi_class_results.pickle')
ml_dictpath = os.path.join(dirname, '..', 'models','cnn', 'multi_label_results.pickle')

    
def ml_evaluate_config(config):
    
    model_name = config['model_name']
    dataset = config['dataset']
    final_activation = config['final_activation']
    loss = config['loss']
    classes = config['classes']
    storage_path = config['storage_path']
    
    classifier = ml_cnn_classifier(model_name=model_name, 
                                   dataset=dataset, 
                                   final_activation=final_activation, 
                                   loss=loss, 
                                   classes=classes, 
                                   model_path=None,
                                   storage_path=storage_path)
    classifier.create_model()
    
    classifier.run_model(batch_size=config['batch_size'], epochs=config['epochs'])

    return classifier
    
    
    
    


"""

def mc_evaluate_config(model_name, dataset, final_activation, loss, classes, batch_size, epochs):
    if not os.path.isfile(ml_dictpath):
        results = {}
    else:
        with open(mc_dictpath, 'rb') as file:
            results = pickle.load(file)
            
    classifier = mc_cnn_classifier(model_name=model_name, dataset=dataset, final_activation=final_activation, loss=loss, classes=classes)
    classifier.create_model()
    modelfile_name, history = classifier.run_model(batch_size=batch_size, epochs=epochs)
    top1_score, top3_score, top3_predictions = classifier.eval()
    
    
    results[modelfile_name]={'top1_score':top1_score,
                               'top3_score':top3_score,
                               'top3_predictions':top3_predictions,
                               'history':history}
    
    with open(mc_dictpath, 'wb') as file:
        pickle.dump(results, file)
        
    return top1_score, top3_score, top3_predictions, classifier
    
"""
    
def plot_history(history, storage_path):
    if not os.path.isdir(storage_path):
        os.mkdir(storage_path)
    for key in list(history.history.keys()):
        if 'val' in key:
            continue
        plt.plot(history.history[key])
        plt.plot(history.history['val_'+key])
        plt.title('model '+key)
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(storage_path, key+'.png'))
        plt.show()



def main_evaluate(i:int):
    #TODO: Speichere Verteilung der Klassen in Train und Test in Datei
    dirname = os.path.dirname(__file__)
    
    config1 = {'model_name':'vgg16_finetuned',
               'dataset':'pascal_voc_reshaped',
               'final_activation':'sigmoid',
               'loss': 'binary_crossentropy',
               'classes':['person', 'horse'],
               'epochs':150,
               'batch_size':32,
               'storage_path':os.path.join(dirname, '..', 'models', 'cnn')
               }
    
    config2 = {'model_name':'vgg16_finetuned',
               'dataset':'pascal_voc_reshaped',
               'final_activation':'sigmoid',
               'loss': 'binary_crossentropy',
               'classes':['cat', 'diningtable', 'person', 'aeroplane', 'bottle'],
               'epochs':100,
               'batch_size':32,
               'storage_path':os.path.join(dirname, '..', 'models', 'cnn')
               }
    
    config3 = {'model_name':'vgg16_finetuned',
               'dataset':'pascal_voc_reshaped',
               'final_activation':'sigmoid',
               'loss': 'binary_crossentropy',
               'classes':[],
               'epochs':100,
               'batch_size':32,
               'storage_path':os.path.join(dirname, '..', 'models', 'cnn')
               }
    
    config4 = {'model_name':'vgg16',
               'dataset':'pascal_voc_reshaped',
               'final_activation':'sigmoid',
               'loss': 'binary_crossentropy',
               'classes':None,
               'epochs':150,
               'batch_size':32,
               'storage_path':os.path.join(dirname, '..', 'models', 'cnn')
               }
    
    classifier_configs = [config1, config2, config3, config4]
    
    dt = datetime.now().strftime('%d_%m_%Y-%H-%M')
    storage_path = os.path.join(dirname, '..', 'models', 'cnn', dt)
    os.mkdir(storage_path)
    config = classifier_configs[i]
    config['storage_path']=storage_path
    classifier = ml_evaluate_config(config)
    try:
        history = classifier.history
    except Exception:
        pass
    
    model = classifier.model
    
    with open(os.path.join(storage_path, 'model_history'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        
    with open(os.path.join(storage_path, 'model_summary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()
            
    with open(os.path.join(storage_path, 'classifier_config.pickle'), 'wb') as f:
        pickle.dump(config, f)
        
    plot_history(history, os.path.join(storage_path, 'plots'))
            
            
    return classifier


def main_load_model(model_dir:Path):
    with open(os.path.join(model_dir, 'classifier_config.pickle'), 'rb') as config_file:
        config = pickle.load(config_file)
    
    print(config)
    model_name = config['model_name']
    dataset = config['dataset']
    final_activation = config['final_activation']
    loss = config['loss']
    classes = config['classes']
    
    model_path = None 
    
    for file in os.listdir(model_dir):
        if ".h5" in file:
            model_path = os.path.join(model_dir, file)
        else:
            continue
    print(model_path)
    classifier = ml_cnn_classifier(model_name=model_name, 
                                   dataset=dataset, 
                                   final_activation=final_activation, 
                                   loss=loss, 
                                   classes=classes, 
                                   model_path=model_path,
                                   storage_path=None)
    classifier.create_model()
    classifier.run_model(0,0)

    return classifier

if __name__ == '__main__':
    classifier = main_evaluate(i=2)
    #classifier = main_load_model(os.path.join(dirname, '..', 'models', 'cnn', '27_06_2020-21-07'))
    classifier.pred(10)
            

