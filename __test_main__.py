from src.plotting.help_func import *
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


model = keras.models.load_model('pretrained_models/cifar10_model_3_multiclass_12_06_2020-23.h5')
# model = keras.models.load_model('pretrained_models/cnn_model3_soft.h5')
# model = keras.models.load_model('pretrained_models/cnn_model3.h5')
model.summary()

idx = np.random.randint(0,10000)
# idx = 5000
pred(model, idx, 'test', eps=0.2, gamma=0.25)




