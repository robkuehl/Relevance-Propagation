from src.rel_prop.help_func import *
from src.rel_prop.rel_prop_grad import rel_prop
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


model = keras.models.load_model('pretrained_models/LeNet5_cifar.h5')
model.summary()

idx = 190

pred(model, idx)


