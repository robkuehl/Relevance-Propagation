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

from src.models.multilabel_cnn import ml_cnn_classifier
from src.models.multiclass_cnn import mc_cnn_classifier
from src.rel_prop.rel_prop import rel_prop
from src.rel_prop.help_func import MidpointNormalize


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
    

def run_rel_prop(model, index, images, eps, gamma):

    image = images[index]
    label = classifier.test_labels[index]
    dataset = 'pascal_test'

    prediction = model.predict(np.array(image, dtype=np.dtype(float)))[0]

    print(f'Correct Label: {label}\n')

    print(f'Network Decision: {prediction}')

    timestamp = time.strftime('%d-%m_%Hh%M')

    label_indices = np.arange(0,len(label))[label == 1]

    for idx in label_indices:
        persist_string = f'{dataset}_{index}_{timestamp}_class_{idx}'

        img = np.array([image.astype(float)])
        mask = np.zeros(len(label), dtype=np.dtype(float))
        mask[idx] = 1.
        mask = tf.constant(mask, dtype=tf.float32)

        plt.subplot(2, 3, 1)
        plt.title(f'{dataset}_{index}')
        fig = plt.imshow(image)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.subplot(2, 3, 2)
        plt.title(f'LRP-0')
        relevance = rel_prop(model, img, mask, eps=0, gamma=0)
        fig = plt.imshow(relevance[0], cmap='seismic',
                   norm=MidpointNormalize(midpoint=0, vmin=relevance[0].min(), vmax=relevance[0].max()))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.subplot(2, 3, 4)
        plt.title(f'LRP-ε (ε={eps})')
        relevance = rel_prop(model, img, mask, eps=eps, gamma=0)
        fig = plt.imshow(relevance[0], cmap='seismic',
                   norm=MidpointNormalize(midpoint=0, vmin=relevance[0].min(), vmax=relevance[0].max()))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.subplot(2, 3, 5)
        plt.title(f'LRP-γ (γ={gamma})')
        relevance = rel_prop(model, img, mask, eps=0, gamma=gamma)
        fig = plt.imshow(relevance[0], cmap='seismic',
                   norm=MidpointNormalize(midpoint=0, vmin=relevance[0].min(), vmax=relevance[0].max()))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.subplot(2, 3, (3, 6))
        plt.title(f'LRP-Composite \neps = {2*eps}\ngamma = {2*gamma}')
        relevance = rel_prop(model, img, mask, eps=2*eps, gamma=2*gamma, comb=True)
        fig = plt.imshow(relevance[0], cmap='seismic',
                   norm=MidpointNormalize(midpoint=0, vmin=relevance[0].min(), vmax=relevance[0].max()))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.savefig('figures/' + persist_string)
    plt.show()


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
    model = classifier.model
    images = classifier.test_images
    index = random.randint(0, images.shape[0])
    eps = 0.25
    gamma = 0.25
    
    run_rel_prop(model, index, images, eps, gamma)
