#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 09:42:09 2020

@author: robin
"""
import os
import pickle
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

import tensorflow as tf
from datetime import datetime
from pathlib import Path

from src.models.multilabel_cnn import ml_cnn_classifier

# Pfad zu dieser Python Datei. Wird genutzt um Speicherort für die Modelle festzulegen.
dirname = os.path.dirname(__file__)

# Methode um die Loss Funktion und die verschiedenen Metriken des Trainings zu plotten
def plot_history(history, storage_path):
    if not os.path.isdir(storage_path):
        os.makedirs(storage_path)
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

"""
Klasse zum Auswerten von Multilabel CNN Archtiketuren für den Pascal Voc Datensatz
- Methoden:
    :add_config: Hinzufügen einer Konfiguration für den Multilabel Classifier. Beispielhafte Configs in self.configs
    :evaluate_config: Erstellt einen Classifier für eine gewählte Cnonfig aus der Liste, trainiert ein Modell, wertet es aus und speichert die Ergebnisse unter dem angegebenen Pfad.
    :load_model: Lädt unter Angabe eines Ordner Namens ein bereits trainiertes Modell
"""
class Pascal_Evaluator():
    
    def __init__(self):
    
        self.configs = [
            
            {
                'model_name':'vgg16_finetuned',
                'dataset':'pascal_voc_reshaped',
                'final_activation':'sigmoid',
                'loss': 'binary_crossentropy',
                'classes':['person', 'horse'],
                'epochs':150,
                'batch_size':32,
                'storage_path':None
            },
        
            {
                'model_name':'vgg16_finetuned',
                'dataset':'pascal_voc_reshaped',
                'final_activation':'sigmoid',
                'loss': 'binary_crossentropy',
                'classes':['cat', 'diningtable', 'person', 'aeroplane', 'bottle'],
                'epochs':100,
                'batch_size':32,
                'storage_path':None
            },
            
            {
                'model_name':'vgg16_finetuned',
                'dataset':'pascal_voc_reshaped',
                'final_activation':'sigmoid',
                'loss': 'binary_crossentropy',
                'classes':[],
                'epochs':100,
                'batch_size':32,
                'storage_path':None
            },
            
            {
                'model_name':'vgg16',
                'dataset':'pascal_voc_reshaped',
                'final_activation':'sigmoid',
                'loss': 'binary_crossentropy',
                'classes':None,
                'epochs':150,
                'batch_size':32,
                'storage_path':None
            }
            
        ]
        
    def add_config(self, config):
        self.configs.append(config)
    
    
    # evaluate_config gibt eine Classifier mit einem trainierten CNN zurück
    def evaluate_config(self, config_nb: int):
        dt = datetime.now().strftime('%d_%m_%Y-%H-%M')
        storage_path = os.path.join(dirname, '..', 'models', 'cnn','pascal', dt)
        os.makedirs(storage_path)
        config = self.configs[config_nb]
        config['storage_path']=storage_path
        
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
        
        try:
            history = classifier.history
        except Exception:
            pass
        
        with open(os.path.join(storage_path, 'model_history'), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
            
        with open(os.path.join(storage_path, 'model_summary.txt'), 'w') as f:
            with redirect_stdout(f):
                classifier.model.summary()
                
        with open(os.path.join(storage_path, 'classifier_config.pickle'), 'wb') as f:
            pickle.dump(config, f)
            
        plot_history(history, os.path.join(storage_path, 'plots'))
                
                
        return classifier
    
    
    # load_model gibt eine Classifier mit einem trainierten CNN zurück
    def load_model(self, folder_name):
        model_dir = os.path.join(dirname, '..', 'models', 'cnn', 'pascal', folder_name)
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
    
    
    

# TODO: Auswertung von Multiclass Architekturen
class Multiclass_Evaluator():
    

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
    









    
            

