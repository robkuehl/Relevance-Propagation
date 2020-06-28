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

from src.plotting.plot_funcs import plot_rel_prop

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# # if GPU is used
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass



dirname = os.path.dirname(__file__)
mc_dictpath = os.path.join(dirname, '..', 'models','cnn', 'multi_class_results.pickle')
ml_dictpath = os.path.join(dirname, '..', 'models','cnn', 'multi_label_results.pickle')

    
def ml_evaluate_config(model_name, dataset, final_activation, loss, classes, batch_size, epochs, model_path):
            
    classifier = ml_cnn_classifier(model_name=model_name, dataset=dataset, final_activation=final_activation, loss=loss, classes=classes, model_path=model_path)
    classifier.create_model()
    modelfile_name, history = classifier.run_model(batch_size=batch_size, epochs=epochs)
    eval_df = classifier.eval()
    
    return eval_df, classifier
    
    
    
    


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
    

def run_rel_prop(classifier, eps, gamma, index):
    # index = random.randint(0, classifier.test_images.shape[0])
    model = classifier.model
    print(len(classifier.test_images))
    image = classifier.test_images[index]*1.0
    label = classifier.test_labels[index]
    dataset = 'pascal_test'

    prediction = model.predict(np.asarray([image]))[0]

    print(f'Correct Label: {label}\n')

    print(f'Network Decision: {prediction}')

    timestamp = time.strftime('%d-%m_%Hh%M')

    label_indices = np.arange(0,len(label))[label == 1]
    label_indices = [0]
    for idx in label_indices:

        titles = []
        relevances = []

        persist_string = f'{dataset}_{index}_{timestamp}_class_{idx}'

        img = np.array([image])
        mask = np.zeros(len(label), dtype=np.dtype(float))
        mask[idx] = 1.
        mask = tf.constant(mask, dtype=tf.float32)

        # # LRP-0
        # titles.append('LRP-0')
        # relevance = rel_prop(model, img, mask)
        # relevances.append(relevance)
        #
        # # LRP-eps
        # titles.append(f'LRP-ε (ε={eps} * std)')
        # relevance = rel_prop(model, img, mask, eps=eps)
        # relevances.append(relevance)
        #
        # # LRP-gamma
        # titles.append(f'LRP-γ (γ={gamma})')
        # relevance = rel_prop(model, img, mask, gamma=gamma)
        # relevances.append(relevance)
        #
        # # LRP-composite
        # titles.append(f'LRP-Composite \neps = {2*eps}\ngamma = {2*gamma}')
        # relevance = rel_prop(model, img, mask, eps=2*eps, gamma=2*gamma, comb=True)
        # relevances.append(relevance)

        # z+
        titles.append('z+')
        relevance = rel_prop(model, img, mask, z_pos=True)
        relevances.append(relevance)

        relevances = tuple(zip(titles, relevances))

        plot_rel_prop(image, relevances, persist_string, True)


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    model_name='vgg16'
    dataset='pascal_voc_reshaped'
    final_activation='sigmoid'
    loss= 'binary_crossentropy'
    classes=['person', 'horse']
    model_path = os.path.join(dirname, '..', 'models','cnn', 'pascal_voc_reshaped_vgg16_multilabel_23_06_2020-13.h5')
    #model_path=None
    batch_size = 5
    epochs = 20
    eval_df, classifier = ml_evaluate_config(model_name, dataset, final_activation, loss, classes, batch_size, epochs, model_path)
    eps = 0.25
    gamma = 0.4
    # for i in np.random.randint(0,158,20):
    for i in [146]:
        run_rel_prop(classifier, eps, gamma, i)
