from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, BatchNormalization, Dropout
from keras.applications.vgg16 import VGG16

def get_model(model_name: str, input_shape: tuple, output_shape: int, final_activation: str):
    
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
        
    
    if model_name == 'ml_mastery':
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
        
    if model_name == 'vgg16':
        vgg16_model = VGG16()
        model = Sequential()
        for layer in vgg16_model.layers[:-1]:
            model.add(layer)
        for layer in model.layers:
            if type(layer) != Dense:
                layer.trainable=False
        model.add(Dense(output_shape, activation=final_activation))
        
      
    return model