from src.rel_prop.min_max_rel_model import MinMaxModel, Nested_Regressor
from src.models.Binary_Mnist_Model import Montavon_Classifier
from src.rel_prop.min_max_rel_model import Nested_Regressor

mc = Montavon_Classifier(class_nb=8, load_model=True)
mc.set_data(test_size=0.2)
mc.set_model()
mc.model.summary()
mc.fit_model(epochs=300, batch_size=32)

minmax = MinMaxModel(classifier=mc)

nr = Nested_Regressor(input_shape=(28,28), use_bias=True, neuron_index=0)
    
    


