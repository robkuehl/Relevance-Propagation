from src.models.Binary_Mnist_Model import Montavon_Model

montavon_model = Montavon_Model(class_nb=8)
montavon_model.set_data(test_size=0.25)
montavon_model.set_model()
montavon_model.fit_model(epochs=100, batch_size=32)
print("Accuracy: {}".format(montavon_model.evaluate(batch_size=32)))
print("Non-trivial accuracy: {}".format(montavon_model.non_trivial_accuracy()))


