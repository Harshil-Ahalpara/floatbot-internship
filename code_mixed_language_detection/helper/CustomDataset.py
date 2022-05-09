import torch
from torch.utils.data import Dataset
import json
import re

class CustomDataset(Dataset):
  def __init__(self, dataframe, tokenizer, testing=False):
    self.tokenizer = tokenizer
    self.data = dataframe
    self.comment_text = dataframe.sentence
    self.targets = dataframe.label.apply(lambda x:json.loads(x))
    self.testing = testing

  def __len__(self):
    return len(self.comment_text)
  
  def __getitem__(self, index):
    sentence = str(self.comment_text[index])
    if self.testing:
      sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence = " ".join(sentence.split())

    inputs = self.tokenizer.encode(sentence)
    ids = inputs.ids
    mask = inputs.attention_mask

    return{
        'ids': torch.tensor(ids, dtype=torch.long),
        'mask': torch.tensor(mask, dtype=torch.long),
        'targets': torch.tensor(self.targets[index], dtype=torch.float)
    }