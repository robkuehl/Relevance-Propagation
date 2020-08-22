from src.models.Binary_Mnist_Model import Montavon_Classifier
from src.rel_prop.rel_prop_min_max_adjusted import run_rel_prop
import tensorflow
from os.path import join as pathjoin
from os.path import isfile
import os
import numpy as np
import pickle

# Das Min-Max-Modell ist hard-coded für das Modell im Montavon Classifier
# Wir berechnen Relevances für Neuronen im 2. und 3. Dense Layer mit der z+ Regel
#   - 2. Dense Layer: Relevances für das Training der Approxximationsmodelle
#   - 3. Dense Layer: Relevances für das Training des Bias im Approximationsmodell
# Anschließend werden die Daten in .npy Dateien gespeichert und beim nächsten Aufruf der Funktion automatisch geladen


def get_higher_relevances(classifier:Montavon_Classifier, recalc_rel:bool, use_higher_rel:bool):
    """Berechnen der Relevances aus dem 2. und 3. Dense Layer des Monatvon Classifiers
    Args:
        classifier (Montavon_Classifier): Objekt der Klasse Monatvon Classifier mit gefittetem Modell
        recalc_rel (bool): Falls True werden die Relevances neu berechnet und neu abgespeichert (auch wenn sie lokal als Datei verfügbar sind)
        use_higher_rel (bool): Wir müssen die higher_relevances (3. Dense Layer) nur dann nutzen, wenn wir sie im Bias des Relevance Models nutzen

    Returns:
        true_relevances (numpy array): Relevances des 2. Dense Layers. Pro Neuron ein Zeilenvektor -> Numpy-Matrix mit shape (100 x #Bilder)
        higher_relevances (numpy array): Relevances des 3. Dense Layers. Pro Bild #(Neuronen im 3. Dense Layer) viele Relevances -> Numpy-Matrix mit shape (400 x #Bilder)
        nr_train_images (numpy array): Bilder mit label 1 für die der Montavon Classifier eine korrekte prediction macht. Diese werden für das Training der Regressor Modelle verwnedet
    """
    
    print("Started to collect relevances to train min-max-model!")
    if use_higher_rel:
        print("Info: You decided to use higher relevances for training.")
    else:
        print("Info: You decided not to use higher relevances for training.")
    dirname = os.path.dirname(__file__)
    # Falls nicht existent, erstelle alle notwendigen Ordner
    storage_path = pathjoin(dirname, "..", "..", "data", "min_max_relevances")
    if not os.path.isdir(storage_path):
        os.path.makedirs(storage_path)
    
    # Falls alle notwendigen Dateien vorhanden sind und die Relevances nicht neu berechnet werden sollen, dann lade die Daten aus den Dateien
    if (isfile(pathjoin(storage_path, "true_relevances.npy")) and isfile(pathjoin(storage_path, "higher_relevances.npy")) and isfile(pathjoin(storage_path, "nr_train_images.npy")) and not recalc_rel):
        print("Load relevances to train min-max-model from local directory!")
        with open(pathjoin(storage_path, "true_relevances.npy"), "rb") as file:
            true_relevances = np.load(file)
    
        with open(pathjoin(storage_path, "higher_relevances.npy"), "rb") as file:
            higher_relevances = np.load(file)
            
        with open(pathjoin(storage_path, "nr_train_images.npy"), "rb") as file:
            nr_train_images = np.load(file)
    
    # Andernfalls, berechne sie neu
    else:
        print("You chose to recalculate or no local directory with precalculated relevances. Started to calculate higher relevances for min-max-model.")
        true_relevances = []
        higher_relevances = []
        pos_classified_indices = []
        # Für jedes Bild im Trainingsdatensatz, führe Relevance Propagation mit z+ aus, sofern der Classifier die zu erkennende Klasse richig erkennt
        # und speichere die Relevances des 2. und 3. Dense Layers in Listen. 
        indices = range(0, len(list(classifier.train_images)))
        for index in indices:
            pred = classifier.predict_train_image(index)
            if index%100 == 0:
                print("Calculate relevance for train image with index {}".format(index))
            if pred == 1 and classifier.train_labels[index]==1:
            #if pred == 1:
                pos_classified_indices.append(index)
                relevances = run_rel_prop(model=classifier.model,
                                        test_images=classifier.train_images,
                                        test_labels=classifier.train_labels,
                                        classes=classifier.classes,
                                        index=index, 
                                        prediction=pred)
                r_true = np.asarray(relevances[-3]).reshape(-1,1)
                true_relevances.append(r_true)
                high_rel = np.asarray(relevances[-2]).reshape(-1,1)
                higher_relevances.append(high_rel)
        
        # Wenn LRP für alle Bilder abgeschlossen ist, konvertiere die Listen in Numpy-Matrizen
        higher_relevances = np.column_stack(higher_relevances)
        true_relevances = np.column_stack(true_relevances)
        nr_train_images = np.asarray([classifier.train_images[index] for index in pos_classified_indices])
        
        # Speichere die Ergebnisse als Numpy Dateien ab
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
