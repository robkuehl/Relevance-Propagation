from src.models.Binary_Mnist_Model import Montavon_Classifier
from src.rel_prop.rel_prop import run_rel_prop
import tensorflow
from os.path import join as pathjoin
from os.path import isfile
import os
import numpy as np

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
def get_higher_relevances(classifier:Montavon_Classifier, recalc:bool, use_higher_rel:bool):
    dirname = os.path.dirname(__file__)
    storage_path = pathjoin(dirname, "..", "..", "data", "min_max_relevances")
    if (isfile(pathjoin(storage_path, "true_relevances.npy")) and isfile(pathjoin(storage_path, "higher_relevances.npy"))) and not recalc:
            with open(pathjoin(storage_path, "true_relevances.npy")) as file:
                true_relevances = np.load(file)
        
            with open(pathjoin(storage_path, "higher_relevances.npy")) as file:
                higher_relevances = np.load(file)
    else:
        true_relevances = []
        higher_relevances = []
        indices = range(0, len(list(classifier.train_images)))
        for index in indices:
            pred = classifier.predict_train_image(index)
            if pred == 1 and classifier.train_labels[index]==1:
                relevances = run_rel_prop(model=classifier.model,
                                        test_images=classifier.train_images,
                                        test_labels=classifier.train_labels,
                                        classes=classifier.classes,
                                        eps=0, gamma=0, 
                                        index=index, 
                                        prediction=pred)
                r_true = relevances[-3].reshape(-1,1)
                true_relevances.append(r_true)
                high_rel = relevances[-2].reshape(-1,1)
                higher_relevances.append(high_rel)
            
        higher_relevances = np.column_stack(higher_relevances)
        true_relevances = np.column_stack(true_relevances)
        
        with open(pathjoin(storage_path, "true_relevances.npy")) as file:
            np.dump(true_relevances, file)
            
        with open(pathjoin(storage_path, "higher_relevances.npy")) as file:
            np.dump(higher_relevances, file)
    
    if use_higher_rel:       
        return true_relevances, higher_relevances
    else:
        return true_relevances, None
        

    
    
    
    