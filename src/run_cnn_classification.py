#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 09:42:09 2020

@author: robin
"""
import tensorflow as tf
import os
import json

# if GPU is used
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

from models.multilabel_cnn_classifier import cnn_classifier

dirname = os.path.dirname(__file__)
dictpath = os.path.join(dirname, '..', 'models','cnn', 'results.json')

    
def evaluate_config(model_type, dataset, classification_type):
    if not os.path.isfile(dictpath):
        results = {}
    else:
        with open(dictpath, 'r') as file:
            results = json.load(file)
            
    classifier = cnn_classifier(model_type=model_type, dataset=dataset, classification_type=classification_type)
    classifier.create_model()
    modelfile_name = classifier.run_model(batch_size=100, epochs=100)
    top1_score, top3_score, top3_predictions = classifier.eval()
    
    results[modelfile_name]={'top1_score':top1_score,
                               'top3_score':top3_score,
                               'top3_predictions':top3_predictions}
    
    with open(dictpath, 'w') as file:
        json.dump(results, file)
        
    return top1_score, top3_score, top3_predictions
    
    
if __name__ == '__main__':
    model_type='model_3'
    dataset='cifar10'
    classification_type='multiclass'
    top1_score, top3_score, top3_predictions = evaluate_config(model_type, dataset, classification_type)
    
    
'''i=0
j=0
while(True):
    if results[i][0]=='False':
        classifier.pred(i)
        j+=1
    if j == 10:
        break
    i=i+1'''