Execution order of scripts for this zip file:
1. pip install -r req.txt
2. python create_data.py data/raw_data data/dataset.csv
3. python train.py data/training_data.csv models/trained.bin
4. python inference.py data/testing_data.csv models/1.bin


Raw Data format: Sentences of each languages should be inside the file 'languages_name.txt' with utf8 encoding.
Number of sentences used to train the model in 50,000 sentences per language and the maximum length of a sentence is 200 words.


--> create_data.py
positional arguments:
    dir_name    path to directory that contains .txt files
    dataset_name     path to save the created .csv file
e.g. python create_data.py data/raw_data data/dataset.csv

This script will create csv file from raw txt files, it will build tokenizer and mapping.json for languages.
It will also create training_data.csv with 80% of dataset and testing_data.csv from rest of the datapoints.


--> train.py
positional arguments:
  training_filename     path to training csv file
  save_model_path       path to save the model

optional arguments:
  -e , --epochs     number of epochs, default: 1
  -m , --load_model     trained model for further training, default: None
  -lr , --learning_rate     learning rate, default: 1e-5
  -b , --batch_size       trainig batch_size, default: 6

e.g. python train.py data/training_data.csv models/trained.bin -e 1 -m models/1.bin -lr 4e-4 -b 5


--> inference.py
positional arguments:
    testing_filename    path to testing csv file
    save_model_path     path to trained model
e.g. 

This script will print classification report for each languages.
For example:
              precision    recall  f1-score   support

      nepali       1.00      1.00      1.00         10
    gujarati       1.00      1.00      1.00         10
      telugu       0.80      1.00      0.89         10
     english       1.00      1.00      1.00         10
     punjabi       1.00      1.00      1.00         10
       hindi       1.00      1.00      1.00         10
    malyalam       1.00      1.00      1.00         10
     bengali       1.00      1.00      1.00         10
     marathi       1.00      0.71      0.83         10
       tamil       1.00      0.67      0.80         10
       Other       0.00      0.00      0.00         0

    accuracy                           0.92        100
    macro avg      0.89      0.85      0.87        100
    weighted avg   0.98      0.92      0.94        100
