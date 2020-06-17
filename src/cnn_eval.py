#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 09:42:09 2020

@author: robin
"""
import tensorflow as tf
import os
import pickle

# if GPU is used
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

from models.multilabel_cnn import ml_cnn_classifier
from models.multiclass_cnn import mc_cnn_classifier

dirname = os.path.dirname(__file__)
mc_dictpath = os.path.join(dirname, '..', 'models','cnn', 'multi_class_results.pickle')
ml_dictpath = os.path.join(dirname, '..', 'models','cnn', 'multi_label_results.pickle')

    
def ml_evaluate_config(model_type, dataset, final_activation, loss, classes):
    if not os.path.isfile(ml_dictpath):
        results = {}
    else:
        with open(ml_dictpath, 'rb') as file:
            results = pickle.load(file)
            
    classifier = ml_cnn_classifier(model_type=model_type, dataset=dataset, final_activation=final_activation, loss=loss, classes=classes)
    classifier.create_model()
    modelfile_name, history = classifier.run_model(batch_size=100, epochs=150)
    score, predictions = classifier.eval()
    
    results[modelfile_name]={'score': score,
                             'predictions':predictions,
                             'history':history,
                             'model_summary':classifier.model.summary}
    
    with open(ml_dictpath, 'wb') as file:
        pickle.dump(results, file)
    
    return score, predictions, history, classifier
    

def mc_evaluate_config(model_type, dataset, final_activation, loss, classes):
    if not os.path.isfile(ml_dictpath):
        results = {}
    else:
        with open(mc_dictpath, 'rb') as file:
            results = pickle.load(file)
            
    classifier = mc_cnn_classifier(model_type=model_type, dataset=dataset, final_activation=final_activation, loss=loss, classes=classes)
    classifier.create_model()
    modelfile_name, history = classifier.run_model(batch_size=100, epochs=200)
    top1_score, top3_score, top3_predictions = classifier.eval()
    
    
    results[modelfile_name]={'top1_score':top1_score,
                               'top3_score':top3_score,
                               'top3_predictions':top3_predictions,
                               'history':history,
                               'model_summary':classifier.model.summary}
    
    with open(mc_dictpath, 'wb') as file:
        json.dump(results, file)
        
    return top1_score, top3_score, top3_predictions, classifier
    


if __name__ == '__main__':
    model_type='model_3'
    dataset='cifar10'
    #dataset='voc_reshaped'
    final_activation='sigmoid'
    loss= 'binary_crossentropy'
    #top1_score, top3_score, top3_predictions = evaluate_config(model_type, dataset, classification_type, classes=['person', 'horse'])
    score, predictions, history, classifier = ml_evaluate_config(model_type, dataset, final_activation, loss, classes=['person', 'horse'])
    
    
    
