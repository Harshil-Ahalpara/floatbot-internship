import torch
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import classification_report
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import DataLoader
from helper.CustomDataset import CustomDataset
from helper.MixCodeModel import MixCodeModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_file = sys.argv[1]
model_path = sys.argv[2]
threshold = 0.75
MAX_TOKEN_LEN = 350

# test_file = 'data/test_dataset.csv'
# model_path = 'models/3.bin'

tokenizer = ByteLevelBPETokenizer('helper/tokenizer/vocab.json', 'helper/tokenizer/merges.txt')
tokenizer.enable_padding(length=MAX_TOKEN_LEN)
tokenizer.enable_truncation(max_length=MAX_TOKEN_LEN)

try:
    test_data = pd.read_csv(test_file)
    print("File imported Successfully...")
    model = torch.load(model_path, map_location=device)
    model.eval()
    print("Model loaded Successfully...")
    

except Exception as e:
    # print("Error in importing csv file or loading model...")
    print(e)
    exit()

print("Evaluating ...")
test_data = test_data[:50]
test_languages_list = list(set(test_data.language.tolist()))
test_languages_list.append("Other")
test_set = CustomDataset(test_data, tokenizer, testing=True)
testing_loader = DataLoader(test_set, batch_size=6)

actual_labels = []
pred_labels = []
with torch.no_grad():
    for _, data in enumerate(testing_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)
        outputs = model.predict(ids, mask)
        actual_labels.extend(targets.cpu().detach().numpy().tolist())
        pred_labels.extend(outputs.cpu().detach().numpy().tolist())


pred_labels = np.array(pred_labels) >= threshold
actual_langs = model.get_lang(actual_labels)    
pred_langs = model.get_lang(pred_labels)

class_report = classification_report(actual_langs, pred_langs, 
                                    labels=test_languages_list, 
                                    zero_division=0)

print(class_report)