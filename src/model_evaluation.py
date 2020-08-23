#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In Dieser Datei stellen wir zwei Klassen zur Verfügung, die zum Auswerten der Modell genutzt werden sollen.
Die Klasse Pascal_Evaluator wertet Modell ausschließlich für den Pascal VOC Datensatz aus.
Für die Klasse Multiclass_Evaluator können verschiedene Datensätze gewählt werden.
Für das Trainieren eines neuen Modells oder trainieren mit einem neuen Datensatz muss wie in den gegebenen Beispielen eine Konfiguration hinzugefügt und ausgewählt werden.
Das Modell und die Auswertung werden automatisch in den zugehörigen Ordner abelegt.
Modell können durch Angabe des Ordner Pfades in dem die entsprechenden Datein abgelegt sind geladen werden.
"""
import os
import pickle
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

import tensorflow as tf
from datetime import datetime
from pathlib import Path

from src.models.multilabel_cnn import ml_cnn_classifier
from src.models.multiclass_classifier import mc_classifier

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
        storage_path = os.path.join(dirname, '..', 'models', 'pascal', dt)
        os.makedirs(storage_path)
        config = self.configs[config_nb]
        config['storage_path']=storage_path
        
        model_name = config['model_name']
        dataset = config['dataset']
        final_activation = config['final_activation']
        loss = config['loss']
        classes = config['classes']
        storage_path = config['storage_path']
        
        print("Evaluating the following configurartion:\n")
        for key in config.keys():
            print("{} : {}".format(key, config[key]))
        
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
    
    
    
    
    
    

class Multiclass_Evaluator():
    
    def __init__(self):
    
        self.configs = [
            
            {
                'model_type':'cnn',
                'model_name':'base_model',
                'dataset':'cifar10',
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
        config = self.configs[config_nb]
        storage_path = os.path.join(dirname, '..', 'models', config['dataset'], config['model_name'], dt)
        if not os.path.isdir(storage_path):
            os.makedirs(storage_path)
        config['storage_path']=storage_path
        
        model_type = config['model_type']
        model_name = config['model_name']
        dataset = config['dataset']
        storage_path = config['storage_path']
        
        print("Evaluating the following configurartion:\n")
        for key in config.keys():
            print("{} : {}".format(key, config[key]))
        
        classifier = mc_classifier(
                                    model_type=model_type,
                                    model_name=model_name, 
                                    dataset=dataset,  
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
    
    
    # load_model gibt einen Classifier mit einem trainiertem CNN zurück
    def load_model(self, folderpath: Path):
        model_dir = Path(folderpath)
        with open(os.path.join(model_dir, 'classifier_config.pickle'), 'rb') as config_file:
            config = pickle.load(config_file)
        
        print(config)
        model_type = config['model_type']
        model_name = config['model_name']
        dataset = config['dataset']
        
        model_path = None 
        
        for file in os.listdir(model_dir):
            if ".h5" in file:
                model_path = os.path.join(model_dir, file)
            else:
                continue
        print(model_path)
        classifier = mc_classifier(
                                    model_type=model_type,
                                    model_name=model_name, 
                                    dataset=dataset,  
                                    model_path=model_path,
                                    storage_path=None)
        classifier.create_model()
        classifier.run_model(0,0)

        return classifier
    