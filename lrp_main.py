import random

import tensorflow as tf
import numpy as np

from src.model_evaluation import Pascal_Evaluator, Multiclass_Evaluator
from src.rel_prop.lrp import run_rel_prop
from src.rel_prop.lrp_utils import get_index


# Angabe, welche LRP-Varianten durchgeführt werden sollen
# Zur Auswahl stehen: 'zero', 'eps', 'gamma', 'komposition' und 'plus' für die bekannten Regeln
lrp_variants = ['zero', 'eps', 'gamma', 'komposition', 'plus']

# Angabe, ob Bild mehrere Labels enthalten soll.
# Hinweis: Es kann trotzdem vorkommen, dass das Netz nicht alle Labels erkennt und so LRP auf weniger oder keine
# Klassifizierung anwendet
multilabel = False

# Falls multilabel = True, wähle, wie viele Labels das Bild enthalten soll
n_labels = 1

# Model und Daten werden geladen
p_e = Pascal_Evaluator()
# TODO: Pfad anpassen, sodass das mit der Readme übereinstimmt
classifier = p_e.load_model('alle_Klassen_ohne_BatchNorm')

index = get_index(classifier=classifier, multilabel=multilabel, n_labels=n_labels)

run_rel_prop(classifier, lrp_variants, index=index, eps=0.2, gamma=0.1)
