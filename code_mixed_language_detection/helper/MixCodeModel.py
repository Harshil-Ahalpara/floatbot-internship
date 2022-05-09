import torch
from transformers import RobertaForSequenceClassification
import os
import json
import numpy as np

mappings = json.load(open('helper/mappings.json'))
if not os.path.exists('models'):
    os.mkdir('models')

class MixCodeModel(torch.nn.Module):
    def __init__(self):
        super(MixCodeModel, self).__init__()
        if os.path.exists('models/roberta-local'):
            self.l1 = RobertaForSequenceClassification.from_pretrained('models/roberta-local', output_hidden_states=True)
        else:
            self.l1 = RobertaForSequenceClassification.from_pretrained('roberta-base', output_hidden_states=True)
            self.l1.save_pretrained('models/roberta-local')

        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, len(mappings.keys()))

    def forward(self, input_ids, attention_mask):
        hiddent_state = self.l1(input_ids, attention_mask=attention_mask)['hidden_states']
        output1 = torch.mean(hiddent_state[-1], dim=1).squeeze()
        output2 = self.l2(output1)
        output = self.l3(output2)
        return output

    def predict(self, input_ids, attention_mask):
        output = self.forward(input_ids, attention_mask)
        return torch.nn.functional.softmax(output, dim=1)
    
    def get_lang(self, array):
        lang_list = []
        for l in array:
            index = np.array(l).argmax().item()
            if l[index]:
                for lang, id in mappings.items():
                    if id == index:
                        lang_list.append(lang)
                        break
            else:
                lang_list.append("Other")
        return lang_list
