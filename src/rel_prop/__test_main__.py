from src.rel_prop.help_func import *
from src.rel_prop.rel_prop_grad import rel_prop
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


# model = keras.models.load_model('pretrained_models/cifar10_model_3_multiclass_12_06_2020-23.h5')
model = keras.models.load_model('pretrained_models/cnn_model3_soft.h5')
model.summary()

idx = 234

pred(model, idx, 'test')


