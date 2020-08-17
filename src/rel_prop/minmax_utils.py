from src.models.Binary_Mnist_Model import Montavon_Classifier
from src.rel_prop.rel_prop_min_max_adjusted import run_rel_prop
import tensorflow
from os.path import join as pathjoin
from os.path import isfile
import os
import numpy as np
import pickle

# NEU FUER DAS MIN-MAX-MODELL: erzeuge Relevances für Neuronen im 2. und 3. Dense Layer mit z+ Regel
#        - 2. Dense Layer: Training von Approxximationsmodell
#        - 3. Dense Layer: Bias im Approximationsmodell
#    Momentan noch Hardgecoded fuer das Sec. III Modell

"""
Speichern der Relevances vom 2. Layer in numpy Datei
    - pro Neuron ein Zelen-Vektor mit Relevance für jedes Bild -> 1 Array
    - speichern als Numpy Matrix shape = (#Neuronen x Bilder)
Speichern der Relevances vom 3. Layer 
    - pro Bild #(Neuronen im vorletzten Layer) viele Relevancen
    - speichere Matrix mit shape = (#Neuronen x Bilder)
Abfragen ob Laden oder erzeugen

"""
def get_higher_relevances(classifier:Montavon_Classifier, recalc_rel:bool, use_higher_rel:bool):
    print("Started to collect relevances to train min-max-model!")
    print("Info: You decided not to use higher relevances for training.")
    dirname = os.path.dirname(__file__)
    storage_path = pathjoin(dirname, "..", "..", "data", "min_max_relevances")
    if not os.path.isdir(storage_path):
        os.path.makedirs(storage_path)
        
    if (isfile(pathjoin(storage_path, "true_relevances.npy")) and isfile(pathjoin(storage_path, "higher_relevances.npy")) and isfile(pathjoin(storage_path, "nr_train_images.npy")) and not recalc_rel):
        print("Load relevances to train min-max-model from local directory!")
        with open(pathjoin(storage_path, "true_relevances.npy"), "rb") as file:
            true_relevances = np.load(file)
    
        with open(pathjoin(storage_path, "higher_relevances.npy"), "rb") as file:
            higher_relevances = np.load(file)
            
        with open(pathjoin(storage_path, "nr_train_images.npy"), "rb") as file:
            nr_train_images = np.load(file)
    else:
        print("No local directory with precalculated relevances. Started to calculate higher relevances for min-max-model.")
        true_relevances = []
        higher_relevances = []
        pos_classified_indices = []
        indices = range(0, len(list(classifier.train_images)))
        for index in indices:
            pred = classifier.predict_train_image(index)
            if index%100 == 0:
                print("Calculate relevance for test image with index {}".format(index))
            if pred == 1 and classifier.train_labels[index]==1:
                pos_classified_indices.append(index)
                relevances = run_rel_prop(model=classifier.model,
                                        test_images=classifier.train_images,
                                        test_labels=classifier.train_labels,
                                        classes=classifier.classes,
                                        eps=0, gamma=0, 
                                        index=index, 
                                        prediction=pred)
                r_true = np.asarray(relevances[-3]).reshape(-1,1)
                true_relevances.append(r_true)
                high_rel = np.asarray(relevances[-2]).reshape(-1,1)
                higher_relevances.append(high_rel)
            
        higher_relevances = np.column_stack(higher_relevances)
        true_relevances = np.column_stack(true_relevances)
        nr_train_images = np.asarray([classifier.train_images[index] for index in pos_classified_indices])
        
        print("Done calculating relevances. Now save relevances into numpy file!")
        with open(pathjoin(storage_path, "true_relevances.npy"), "wb") as file:
            np.save(file, true_relevances)
            
        with open(pathjoin(storage_path, "higher_relevances.npy"), "wb") as file:
            np.save(file, higher_relevances)
            
        with open(pathjoin(storage_path, "nr_train_images.npy"), "wb") as file:
            np.save(file, nr_train_images)
            
    
    if use_higher_rel:       
        return true_relevances, higher_relevances, nr_train_images
    else:
        return true_relevances, None, nr_train_images
        

    
    
    
    