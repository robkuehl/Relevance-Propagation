To run the code:    - add the src folder to pythonpath
                    - you may use the jupyter notebook 'main'

Beschreibung der Projektstruktur:

├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-rk-first-model-testing`.
│
├── references         <- Data paper, relevant informations, created explenations
│
├── presentations      <- Generated presentations as Jupyter, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in presentations
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
│   ├── process       <- Scripts to turn raw data into data for training e.g. encode labels
│   │
│   ├── models         <- Scripts to train models and then use trained models to make predictions
│   │   │           
│   │   ├── e.g predict_model.py
│   │   └── e.g train_model.py
│   │
│   └── rel_prop  <- Scripts to run relevance propagation for models and images
│       └── e.g rel_prop.py
│
└── gitignore-file