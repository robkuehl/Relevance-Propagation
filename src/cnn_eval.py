#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 09:42:09 2020

@author: robin
"""
import tensorflow as tf
import os
import pickle
from models.multilabel_cnn import ml_cnn_classifier
from models.multiclass_cnn import mc_cnn_classifier

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

    
def ml_evaluate_config(model_name, dataset, final_activation, loss, classes, batch_size, epochs):
            
    classifier = ml_cnn_classifier(model_name=model_name, dataset=dataset, final_activation=final_activation, loss=loss, classes=classes)
    classifier.create_model()
    modelfile_name, history = classifier.run_model(batch_size=batch_size, epochs=epochs)
    eval_df = classifier.eval()
    
    return eval_df, history, classifier
    




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
    





if __name__ == '__main__':
    model_name='vgg16'
    #dataset='cifar10'
    dataset='pascal_voc_reshaped'
    final_activation='sigmoid'
    loss= 'binary_crossentropy'
    classes=['person', 'horse']
    batch_size = 5
    epochs = 100
    #top1_score, top3_score, top3_predictions = evaluate_config(model_name, dataset, classification_type, classes=['person', 'horse'])
    eval_df, history, classifier = ml_evaluate_config(model_name, dataset, final_activation, loss, classes, batch_size, epochs)
    
    
    
