Beschreibung der Projektstruktur:

├── README.md          <- The top-level README on how to this project.
├── data
│   ├── processed      <- Data for training of models (e.g. rehsaped pascal)
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── e.g. make_dataset.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make predictions
│   │           
│   ├── model_evaluation.py <- Methode zum Auswerten von Modellen
|   |
│   └── rel_prop        <- Scripts to run relevance propagation for models and images
│       └── e.g rel_prop.py
|
├── lrp_main.py         <- Main Methode zum Ausführen von Relevacne Propagation
|
├── minmax_main.py      <- Main Methode zum Ausführen des Min-Max-Modells
|
|
├── minmax_results      <- Ordner in der Bilder gespeichert werden, die mit minmax_main.py generiert werden
│
└── gitignore-file




Anwendung Min-Max-Model
Das Min-Max Modell kann auschließlich für den Montavon Classifier auf dem MNIST Datensatz verwendet werden.
Die higher relevacnes (Input für Bias Term) und die trainierten Hilfsmodelle werden automatisch geladen, sofern sie bereits vorhanden sind.
Mit nb_images kann die gewünschte Anzahl zu erzeugender Bilder gewählt werden.



Anwendung LRP
1. Wahl der Parameter in lrp_main.py
	- lrp_variants: Ein Array, das die Namen der gewünschten LRP-Regeln enthalten soll
	- multilabel: Boolean, der angibt, ob ein Bild mit mehreren Labeln gewählt werden soll
	- n_labels: Anzahl der vorhandenen Labeln im Bild

2. lrp_main.py ausführen
