import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
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

"""Plotte nur das Mnist Bild"""
def plot_mnist_image(image):
    fig = px.imshow(image, color_continuous_scale=px.colors.sequential.gray_r)
    fig.show()



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
Das Modell zur Demonstration des Min-Max Prinzips aus Montavon et al.
Besteht aus: 
Flattened Input
Dense Layer mit 400 Detektionsneuronen, relu aktivierung
Pooling Operation: 400 neuronen jeweils 4 summieren -> 100 neuronen Output
Dense Layer mit 400 Detektionsneuronen, relu aktivierung
Globales sum-pooling zu einem outputneuron. Output sollte > 0 sein, falls Zahl erkannt wurde, 0 sonst
""" 
def getMnistModel(input_shape = (28,28)):
    #Eingebaute Funktion, die die Uebergangsmatrix mit 1en initialisiert.
    ones_initializer = tf.keras.initializers.Ones()
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(400, activation='relu', use_bias = False))
    #Layer wird erstmal normal initialisiert, ones_initializer nicht unbedingt notwendig, da die matrix eh ersetzt wird, spart aber ggf. Rechenaufwand.
    custom_pooling = Dense(100, activation = 'relu', use_bias = False, kernel_initializer = ones_initializer)
    #Gewichte sollen nicht veraendert werden
    custom_pooling.trainable=False
    model.add(custom_pooling)
    model.add(Dense(400, activation='relu', use_bias = False))
    #Gleiches wie oben, kernel wird mit 1en initialisiert und nicht trainierbar -> sum-pooling
    sum_pooling = Dense(1, activation = 'relu', use_bias = False, kernel_initializer = ones_initializer)
    sum_pooling.trainable = False
    model.add(sum_pooling)
    #Uebergangsmatrix bei sum_pooling aendern: Normalerweise bekommt set_weights eine Liste mit W und b -> hier use_bias=False
    list_of_weights = [np.transpose(getSumPoolingWeights())]
    model.layers[2].set_weights(list_of_weights)
    return model


""" 
    Theos LRP Klasse aus dem ersten Vortrag.
    TODO Hier muessten die neuen LRP-Methoden angebunden werden!
    Neu hinzugekommen sind Methoden, um die hoeheren Relevanzen (Mittlerer Layer und vorletzter Layer im Sec III Modell) zu extrahieren
"""
class relevance_propagation:


    def calc_r(self, r: np.ndarray, output: np.ndarray, weights: np.ndarray, eps: int = 0, beta: int = None):

        #Siehe LaTeX erster Vortrag
        nominator = np.multiply(np.transpose(output),
                                weights)
        #ggf. beta-Regel anwenden (Deep Taylor z+)
        if beta is not None:
            if eps:
                print('+++ERROR+++')
                print('Choose either EPS or BETA, not both!')
                print('+++ERROR+++')
                sys.exit()

            zero = np.zeros(nominator.shape)
            z_pos = np.maximum(zero, nominator)
            z_neg = np.minimum(zero, nominator)

            denominator_pos = np.sum(z_pos, axis=0)
            denominator_neg = np.sum(z_neg, axis=0)

            fraction_pos = np.divide(z_pos, denominator_pos)
            fraction_neg = np.divide(z_neg, denominator_neg)

            fraction = (1 - beta) * fraction_pos + beta * fraction_neg

        else:
            try:
                denominator = np.matmul(output,
                                    weights)
            except ValueError:
                print("Dimension error in calc_r, outputs dimension {}, weights dimension {}".format(output.shape, weights.shape))
            

            if eps:
                denominator = denominator + eps * np.sign(denominator)
            # 0.0 im Nenner darf nicht sein, ersetze durch Kleine Zahl
            if 0.0 in denominator:
                denominator[denominator==0.0]=0.00000001

            fraction = np.divide(nominator, denominator)
            #Falls doch noch NaN Werte auftreten sollten...
            if np.isnan(fraction).any():
                print("nan values in fraction, shape {} and values \n{}".format(fraction.shape, fraction))
                

        r_new = np.dot(fraction, r)

        return r_new


    # Funktion für Relevance Propagation (Beliebige Anzahl Schichten)
    def rel_prop(self, model: tf.keras.Sequential, input: np.ndarray, eps: float = 0.0, beta: float = None) -> np.ndarray:

        # Hilfsmodel zum Extrahieren der Outputs des Hidden Layers
        extractor = tf.keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

        features = extractor(np.array([input]))
        #extract number of features
        nFeatures = len(features)
        outputs = []
        weightMatrices = []
        for layer in range(nFeatures):
            outputs.append(features[layer].numpy())
        # decrease nFeatures to collect weight matrices (one less than no. of features)
        nFeatures-=1
        weights = []
        for betweenLayer in range(nFeatures):
            weightMatrices.append(model.weights[betweenLayer].numpy())
        # Equivalent to R^{(l+1)}, initiated with the output
        rel_prop_vector_2 = np.transpose(outputs[nFeatures])
        # again, decrease nFeatures by one to get access to the list indices -> Backwards calculation of input Relevance vector
        nFeatures-=1
        while(nFeatures>=0):
            #rel_prop_vector_1 is equivalent to R^{(l)} from the notebook
            rel_prop_vector_1= self.calc_r(r=rel_prop_vector_2, 
                output=outputs[nFeatures],
                weights=weightMatrices[nFeatures],
                eps=eps,
                beta=beta)
            nFeatures-=1
            rel_prop_vector_2 = rel_prop_vector_1
            if np.isnan(rel_prop_vector_1).any():
                print("nan values in relevance_propagation vector shape {} and values \n{}".format(rel_prop_vector_1.shape, rel_prop_vector_1))
                return np.zeros(input.shape)

        # Finally, output of the relevance propagation is the same dimension as the flattened input vector, 
        # reshape it to original dimensions
        relevance = np.reshape(rel_prop_vector_1, input.shape)

        return relevance

    
   
    """
        NEU FUER DAS MIN-MAX-MODELL: Extrahiere die Relevanzen aus den hoeheren Schichten.
        Momentan noch Hardgecoded fuer das Sec. III Modell
    """
    def get_higher_relevances(self, model: tf.keras.Sequential, input, eps : float = 0.0, beta : float = None):
         # Hilfsmodel zum Extrahieren der Outputs des Hidden Layers
        extractor = tf.keras.Model(inputs=model.inputs,
                                outputs=[layer.output for layer in model.layers])
        # Hilfsmodel berechnet Output der einzelnen Schichten für gegebenen Input
        features = extractor(np.array([input]))

        # wg sum_pooling ist die Relevanz {R_l} = x_l
        high_relevances = features[3].numpy()
        
        # Outputs der einzelnen Schichten
        output_midlayer = features[2].numpy()

        # Berechnung der Relevanzen des mittleren Layers
        mid_relevances = self.calc_r(r=np.transpose(high_relevances),
                    output=output_midlayer,
                    weights=model.weights[2].numpy(),
                    eps=eps,
                    beta=beta)
    
        return high_relevances, mid_relevances

#END class relevance_propagation


############################################################################################################
########### HILFSMETHODEN (aktuell nicht genutzt) ##########################################################
############################################################################################################

"""
    Plotte Bild und Standard LRP
"""
def plot_rel_prop(image, model):
    plot_mnist_image(image)               
    rel = relevance_propagation().rel_prop(model, image)
    plot_rel(rel)
    print(test_labels[i])   

"""
    Fuer das Uebergebene Model, Plotte die ersten <num_images> korrekt erkannten Bilder und Zeige jeweils Bild und LRP
"""
def plot_images_with_rel(test_images, test_labels, model, nb_class):
    # stelle ein Objekt der Klasse Relevance Propagation
    rp = relevance_propagation()
    num_images = 1
    # Führe Relevance Propagation für die ersten <num_images> Bilder der Klasse nb_class aus, die der Classifier korrekt erkennt
    j=0
    i=0
    while j<num_images:
        if test_labels[i]==nb_class:
            prediction = model.predict(np.array([test_images[i]]))
       

            if prediction[0][0]>=0.5:
                print("predicted :{} on image:".format(prediction))
                plot_mnist_image(test_images[i])
                j+=1
                image = test_images[i]
                rel = rp.rel_prop(model, image)
                plot_rel(rel)
                print(test_labels[i])
        i+=1