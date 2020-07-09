
#from src.models.section_3_model import relevance_propagation
import tensorflow as tf
"""
TODO Klasse zum Speichern der benoetigten Daten:
Pro Trainingsdaten-Input wird gebraucht:
    - Input
    - Relevance Propagation Vektor der mittleren Schicht ({R_k})
    - Relevance Propagation Vektor der vorletzten Schicht ({R_l})
"""

try:
    from section_3_model import relevance_propagation
except ImportError as error:
    # Include the name and path attributes in output.
    print("error.name: {}".format(error.name))
    print("error.path: {}".format(error.path))



class model_data:

    def __init__(self, dataset, relevance_propagation):
        self.dataset = dataset
        self.relevance_propagation = relevance_propagation
        self.mid_relevances = []
        self.high_relevances = []
        self.model_list = []
   
    def set_data(self,model):
        test_input = self.dataset[0]
        print("test_input shape {}".format(test_input.shape))
        first_test_image = test_input[0]
        print("first_test_image shape {}".format(first_test_image.shape))
        high_relevance, mid_relevance = self.relevance_propagation.get_higher_relevances(model,first_test_image)
        print("extracted the following high and mid-layer relevances, shape high {}, shape mid {}, values high \n{}, \n values mid \n{}".format(high_relevance.shape, mid_relevance.shape, high_relevance, mid_relevance))
        print("extract necessary information from the provided model...")
    
    def load_data_from_pickle(self,path):
        print("loading data from pickle...")
    
    def save_data_to_pickle(self,path):
        print("saving data as pickle...")
#END class model_data


"""
Das eigentliche min_max_model.
Bekommt das keras-model und die Dateien in model_data uebergeben
"""
class min_max_model:
    
    def __init__(self, model, model_data):
        self.model_data = model_data
