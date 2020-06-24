#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 09:42:09 2020

@author: robin
"""
import time
import os
import pickle
import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from models.multilabel_cnn import ml_cnn_classifier
from models.multiclass_cnn import mc_cnn_classifier
from rel_prop.rel_prop import rel_prop
from rel_prop.help_func import MidpointNormalize



# if GPU is used
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass



dirname = os.path.dirname(__file__)
ml_dictpath = os.path.join(dirname, '..', 'models','cnn', 'multi_label_results.pickle')


    
def ml_evaluate_config(model_name, dataset, final_activation, loss, classes, batch_size, epochs, model_path):
            
    classifier = ml_cnn_classifier(model_name=model_name, dataset=dataset, final_activation=final_activation, loss=loss, classes=classes, model_path=model_path)
    classifier.create_model()
    modelfile_name, history = classifier.run_model(batch_size=batch_size, epochs=epochs)
    eval_df = classifier.eval()
    
    
    return eval_df, classifier

    

    

if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    model_name='vgg16'
    dataset='pascal_voc_reshaped'
    final_activation='sigmoid'
    loss= 'binary_crossentropy'
    classes=['person', 'horse']
    #classes = None
    model_path = os.path.join(dirname, '..', 'models','cnn', 'pascal_voc_reshaped_vgg16_multilabel_21_06_2020-15.h5')
    model_path=None
    batch_size = 5
    epochs = 20
    eval_df, classifier = ml_evaluate_config(model_name, dataset, final_activation, loss, classes, batch_size, epochs, model_path)
    eps = 0.25
    gamma = 0.25
    
    #run_rel_prop(classifier, eps, gamma)
 