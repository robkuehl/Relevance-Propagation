from src.models.Binary_Mnist_Model import Montavon_Model


def get_higher_relevances():
    # TODO:
    # NEU FUER DAS MIN-MAX-MODELL: erzeuge Relevances für Neuronen im 2. und 3. Dense Layer mit z+ Regel
    #        - 2. Dense Layer: Training von Approxximationsmodell
    #        - 3. Dense Layer: Bias im Approximationsmodell
    #    Momentan noch Hardgecoded fuer das Sec. III Modell
    
    """
    Speichern der Relevances vom 2. in pickle Datei
        - pro Neuron ein Vektor mit Relevance für jedes Bild -> 1 Array
        - speichern als Numpy Matrix shape = (#Neuronen x Bilder)
    Speichern der Relevances vom 3. Layer 
        - pro Bild #(Neuronen im vorletzten Layer) viele Relevancen
        - speichere Matrix mit shape = (#Neuronen x Bilder)
    Abfragen ob Laden oder erzeugen
    
    """