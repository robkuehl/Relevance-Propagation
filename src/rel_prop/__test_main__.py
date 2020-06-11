from src.rel_prop.help_func import *
from src.rel_prop.rel_prop_grad import rel_prop
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


model = keras.models.load_model('pretrained_models/LeNet5_cifar.h5')
# pred(model, 20)
layer = model.layers[0]
# # print(layer)
# print(layer.weights)
# weights = [tf.clip_by_value(np.multiply(layer.get_weights(), 2)[0], clip_value_min=0, clip_value_max=np.inf), tf.clip_by_value(np.multiply(layer.get_weights(), 2)[1], clip_value_min=0, clip_value_max=np.inf)]
# print(weights)
# # weights = tf.constant(tf.maximum(layer.get_weights()[0], 0))
# # tf.add(layer.get_weights(), weights)
# layer.set_weights(weights)
# print(layer.weights)
# # print(layer.weights)

def rho(layer):
    layer.set_weights([tf.clip_by_value(np.multiply(layer.get_weights(), 2)[0],
                                        clip_value_min=0, clip_value_max=np.inf),
                       tf.clip_by_value(np.multiply(layer.get_weights(), 2)[1],
                                        clip_value_min=0, clip_value_max=np.inf)])
    return layer

print(layer.weights)
layer = rho(layer)
print(layer.weights)
# idx = 100
#
# pred(model, idx)
#
# # img = np.array([x_test[idx].astype(float)])
# #
# # relevance = rel_prop(model, img)
# # relevance = relevance.sum(axis=3)
# # plt.subplot(2,1,1)
# # plt.imshow(relevance[0], cmap='seismic', norm=MidpointNormalize(midpoint=0, vmin=relevance[0].min(), vmax=relevance[0].max()))
# # plt.subplot(2,1,2)
# # plt.imshow(x_test[idx])
# # plt.show()


# for layer in model.layers:
#     img = layer(img)

# print(img)