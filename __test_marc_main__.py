import os
from tensorflow.keras.datasets import mnist
from src.models.section_3_model import *

"""
Testumgebung fuer Min-Max
"""

#Mnist Daten laden
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
data = [train_images, train_labels, test_images, test_labels]

# Definiere die Klasse, für die wir einen binären Classifier trainieren wollen
class_nb = 8

def plot_rel(heatmap):
    imageArray = np.asarray(heatmap)
    #get min and max value and define the bound for heatmap
    min_val = np.amin(imageArray)
    max_val = np.amax(imageArray)
    bound = np.amax(np.array([-1* min_val, max_val]))
    fig = px.imshow(heatmap, color_continuous_scale=px.colors.sequential.Cividis, zmin=-1*bound, zmax=bound)
    fig.show()

def plot_rel_prop_comparison(image, model, mdl_data):
    plot_mnist_image(image)               
    heatmap_z = relevance_propagation().rel_prop(model, image)
    heatmap_z_min_max = mdl_data.rel_prop(image)
    plot_rel(heatmap_z)
    plot_rel(heatmap_z_min_max)
    #print(test_labels[i])  

def get_good_examples(mdl_data, model):
    num_images = 2
    # Führe Relevance Propagation für die ersten <num_images> Bilder der Klasse nb_class aus, die der Classifier korrekt erkennt
    j=0
    i=0
    indices = []
    while j<num_images and i<len(test_images):
        if test_labels[i]==class_nb:
            has_nonzero = model.predict(np.array([test_images[i]]))
            if mdl_data.has_nonzero(test_images[i]):
                if model.predict(np.array(test_images[i]).reshape(1,28,28))[0][0] > 0.5:
                    print("using index {} for relevance_propagation".format(i))
                    print("correct label: {}".format(test_labels[i]))
                    indices.append(i)
                    j+=1
        i+=1

    return indices
     

# Trainiere das neuronale Netz des Classifiers (Code aus Vortrag 1)
cl = get_binary_cl(data=data, dataset='mnist', model_type='load_dense', class_nb=class_nb, epochs = 20)
model = cl.getModel()
inputs = model.inputs
#print("dtype of inputs: {}".format(inputs.dtype))
#plot_images_with_rel(data[2], data[3], model, class_nb)
rp = relevance_propagation()
data[0], data[1] = cl.getBinaryData()
mdl_data = model_data(data, rp)
mdl_data.set_data(model)
mdl_data.load_models()
image_array = get_good_examples(mdl_data,model)
#image_array = [1669,1756]
for image_index in image_array:
    image_to_plot = test_images[image_index]
    print("image_index to plot: {}".format(image_index))
    plot_rel_prop_comparison(image_to_plot,model,mdl_data)