import numpy as np
import plotly.express as px

############################################################################################################
########### RELEVANCE PROPAGATION METHODEN #################################################################
############################################################################################################
""" Plotte nur die Relevance-Propagation Heatmap
    Heatmap Skala wird am betraglich groessten Wert ausgerichtet
"""
def plot_rel(heatmap):
    imageArray = np.asarray(heatmap)
    #get min and max value and define the bound for heatmap
    min_val = np.amin(imageArray)
    max_val = np.amax(imageArray)
    bound = np.amax(np.array([-1* min_val, max_val]))
    fig = px.imshow(heatmap, color_continuous_scale=px.colors.sequential.Cividis, zmin=-1*bound, zmax=bound)
    fig.show()

""" Plotte 3 Grafiken: Das mnist Bild, Min-Max-Modell-LRP und Standard LRP-0 Regel (z-Regel)
"""
def plot_rel_prop_comparison(image, model, mdl_data):
    plot_mnist_image(image)               
    heatmap_z = relevance_propagation().rel_prop(model, image)
    heatmap_z_min_max = mdl_data.rel_prop(image)
    plot_rel(heatmap_z)
    plot_rel(heatmap_z_min_max)

############################################################################################################
########### MODELL AUS SEC. III (MONTAVON) #################################################################
############################################################################################################

"""
Definiert eine Matrix, die inputDim mittels Sum Pooling auf outputDim reduziert
"""
def getSumPoolingWeights(inputDim = 400, outputDim = 100):
    #Bestimme die Anzahl an Neuronen, die auf ein Outputneuron summiert werden
    pool_ratio = int(inputDim /outputDim)
    #TODO Fehler werfen, falls kein int!!
    #Liste zum Speichern der Zeilen der Matrix
    row_list = []
    for row in range(outputDim):
        this_row = np.zeros(inputDim)
        #Zeile soll nur <pool_ratio> viele 1en an der richtigen Position haben
        this_row[pool_ratio * row:pool_ratio*row+pool_ratio]=1.0
        row_list.append(this_row)
    #Gewichtsmatrix setzt sich zusammen aus den erzeugten Zeilen
    weight_matrix = np.asarray(row_list)    
    return weight_matrix



"""
Das Modell aus Sec III in Montavon et al
Besteht aus: 
Flattened Input
Dense Layer mit 400 Detektionsneuronen, relu aktivierung
Pooling Operation: 400 neuronen jeweils 4 summieren -> 100 neuronen Output
Dense Layer mit 400 Detektionsneuronen, relu aktivierung
Globales sum-pooling zu einem outputneuron. Output soll > 0 sein, falls Zahl erkannt wurde, 0 sonst
""" 
def getSection3Model(input_shape = (28,28)):
    #Eingebaute Funktion, die die Uebergangsmatrix mit 1en initialisiert.
    ones_initializer = tf.keras.initializers.Ones()
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(400, activation='relu', use_bias = False))
    #Kernel initializer sorgt dafuer, dass die Gewichtsmatrix die geforderte Pooling Operation realisiert
    custom_pooling = Dense(100, activation = 'relu', use_bias = False, kernel_initializer = ones_initializer)
    #Gewichte sollen nicht veraendert werden
    custom_pooling.trainable=False
    model.add(custom_pooling)
    model.add(Dense(400, activation='relu', use_bias = False))
    #Gleiches wie oben, kernel wird mit 1en initialisiert und nicht trainierbar -> sum-pooling
    sum_pooling = Dense(1, activation = 'relu', use_bias = False, kernel_initializer = ones_initializer)
    sum_pooling.trainable = False
    model.add(sum_pooling)
    #Uebergangsmatrix bei sum_pooling aendern:
    list_of_weights = [np.transpose(getSumPoolingWeights())]
    #print("list of weights [0] shape: {}, [1] shape {}".format(list_of_weights[0].shape, list_of_weights[1].shape))
    model.layers[2].set_weights(list_of_weights)
    return model
