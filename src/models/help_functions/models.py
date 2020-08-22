"""
In dieser Datei sammeln wir verschiedene Modelle (Neuronale Netze), die für das Training und die Relevance Propagation auf den verschiedenen Datensätzen 
verwendet werden können.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, BatchNormalization, Dropout, Activation
from tensorflow.keras.applications.vgg16 import VGG16

def get_cnn_model(model_name: str, input_shape: tuple, output_shape: int, final_activation: str):
    """Laden verschiedener CNN Modellarchitekturen

    Args:
        model_name (str): Definiere welches Modell geladen werden soll. Optionen: 'base_model', 'vgg_simple', 'vgg16_finetuned'
        input_shape (tuple): Inputshape der Trainingsdaten
        output_shape (int): Anzahl der Klassen im Datensatz
        final_activation (str): Aktivierungsfuntion im letzen Layer
    
    Returns:
        model [tensorflow.keras.models.Sequential]: Modell nach gewählter Konfiguration
    """
    

    
    if model_name == 'base_model':
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(256, kernel_size=3, activation='relu', ))
        model.add(Flatten())
        model.add(Dense(40, activation='relu'))
        model.add(Dense(output_shape, activation='sigmoid'))
        
    
    if model_name == 'vgg_simple':
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(output_shape, activation=final_activation))
        
    # Wir laden das bereits trainierte VGG-16 Modell
    # Das originale Modell wurde für eine Multiclass Klassifikation auf dem Imagenet Datensatz trainiert
    # Daher verwenden wir nur die Feature Detektoren des originalen Netzes und trainieren neu hinzugefügte Dense Layer
    if model_name == 'vgg16_finetuned':
        vgg16_model = VGG16()
        model = Sequential()
        for layer in vgg16_model.layers[:-1]:
            if type(layer) != Dense:
                model.add(layer)
        for layer in model.layers:
            layer.trainable=False
        #model.add(BatchNormalization())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.25))
        #model.add(BatchNormalization())
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.25))
        #model.add(BatchNormalization())
        model.add(Dense(output_shape))
        model.add(Activation(final_activation))
        
    return model

