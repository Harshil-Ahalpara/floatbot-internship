# Directories
## data
-It contains data used in training and inferencing.
-Format: (.csv) grammatically wrong sentence, grammatically correct sentence


## models
-On the first run, it will be created if not present and the local copy of t5-base model will be stored in this directory.
-You can later on save the trained models in this directory.


# Files
## req.txt
- pip install -r req.txt


## params.py
-This file contains parameters that will be used in training and inferenceing.


## train.py
### positional arguments:
    training_filename     path to training csv file containing "input" and "target" columns
    save_model_path       path to save the model

### optional arguments:
    -h, --help            show this help message and exit
    -m, --model           previously trained model for further training, default=t5-base

-Example: python train.py data/train_data.csv models/t5_base_v1.1 -m  models/t5_base_v1.0


## inference.py
### positional arguments:
    process       evaluate a csv file or generate corrected sentences(eval/gen)
    model_path    path to the model

### optional arguments:
    -h, --help            show this help message and exit
    -s, --sentence          for only one sentence, default=None
    -f, --file          file containing multiple sentences, default=None

# Example commands:

    evaluating the csv file       python inference.py eval models/t5_base_v1.1  -f data/test_data.csv
    generating single sentence    python inference.py gen models/t5_base_v1.1  -s 'i will go home yesterday.'
    generating file               python inference.py gen models/t5_base_v1.1  -f data/test.txt