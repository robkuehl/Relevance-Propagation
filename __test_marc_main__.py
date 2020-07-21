import os
from tensorflow.keras.datasets import mnist
from src.models.min_max_utils import relevance_propagation
from src.models.min_max_model import get_binary_cl
from src.models.min_max_model import min_max
from src.models.min_max_model import plot_rel_prop_comparison

"""
Testumgebung fuer Min-Max
"""

#Mnist Daten laden
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
data = [train_images, train_labels, test_images, test_labels]

# Definiere die Klasse, für die wir einen binären Classifier trainieren wollen
class_nb = 8


    #print(test_labels[i])  

""" Suche nach Bildern, fuer die sowohl Das Min-Max Modell, als auch das Standard Modell eine sinnvolle Heatmap liefern sollten.
    (Beim Min-Max Modell, schiesse den Input durch jedes einzelne Hilfsnetz und schaue, ob wenigstens eines >0 Werte ausgibt)
"""
def get_good_examples(min_max_mdl, model):
    num_images = 2
    # Führe Relevance Propagation für die ersten <num_images> Bilder der Klasse nb_class aus, die der Classifier korrekt erkennt
    j=0
    i=0
    indices = []
    while j<num_images and i<len(test_images):
        if test_labels[i]==class_nb:
            has_nonzero = model.predict(np.array([test_images[i]]))
            if min_max_mdl.has_nonzero(test_images[i]):
                if model.predict(np.array(test_images[i]).reshape(1,28,28))[0][0] > 0.5:
                    print("using index {} for relevance_propagation".format(i))
                    print("correct label: {}".format(test_labels[i]))
                    indices.append(i)
                    j+=1
        i+=1

    return indices
     

# Trainiere das neuronale Netz des Classifiers (Code aus Vortrag 1)
cl = get_binary_cl(data=data, dataset='mnist', model_type='load_dense', class_nb=class_nb, epochs = 2)
model = cl.getModel()
rp = relevance_propagation()
#Der Datensatz wurde in cl binaerisiert, arbeite mit den binaeren Labels weiter
data[1], data[3] = cl.getBinaryLabels()
min_max_mdl = min_max(data, rp)
min_max_mdl.set_data(model)
#Hier ggf. durch train_models ersetzen, um neu zu trainieren
min_max_mdl.load_models()

#image_array = get_good_examples(mdl_data,model)
image_array = [1669,1756]
for image_index in image_array:
    image_to_plot = test_images[image_index]
    print("image_index to plot: {}".format(image_index))
    plot_rel_prop_comparison(image_to_plot,model,min_max_mdl)