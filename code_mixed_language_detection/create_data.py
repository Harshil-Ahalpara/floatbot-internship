import os
import re
import pandas as pd
import json
import sys
from tokenizers import ByteLevelBPETokenizer


MAX_TOKEN_LEN = 350
train_size = 0.8
raw_data_dir = sys.argv[1]
save_data_path = sys.argv[2]


def create_mapping(file_names):
    l = len(file_names)
    mappings = {}
    count = 0
    for name in file_names:
        mappings[name[:-4]] = count
        count += 1
    return mappings


def get_language_label(f_name, mapping):
    language = f_name[:-4]
    label = [0 for _ in range(len(mapping))]
    label[mapping[language]] = 1
    return language, label


if __name__ == "__main__":

    print("Creating .csv file...\n")
    file_names = os.listdir(raw_data_dir)
    data_list = []
    mapping = create_mapping(file_names)
    json.dump(mapping, open('helper/mappings.json', 'w'))

    for file_name in file_names:
        f = open(os.path.join(raw_data_dir, file_name), 'r', encoding='utf8')
        lines = f.readlines()

        language, label = get_language_label(file_name, mapping)

        for line in lines:
            sentence = line.strip()
            sentence = re.sub(r'[^\w\s]', '', sentence)
            row = (sentence, language, label)
            data_list.append(row)

    dataset = pd.DataFrame(data_list, columns=['sentence','language', 'label'])
    dataset = dataset.sample(frac=1)
    
    training_data = dataset.sample(frac=train_size)
    testing_data = dataset.drop(training_data.index).reset_index(drop=True)
    training_data = training_data.reset_index(drop=True)

    dataset.to_csv(save_data_path, index=False)
    training_data.to_csv(os.path.join(os.path.dirname(save_data_path), 'training_data.csv'), index=False)
    testing_data.to_csv(os.path.join(os.path.dirname(save_data_path), 'testing_data.csv'), index=False)

    print("Creating Vocabulary ...\n")
    vocab_file = open('helper/vocabulary.txt', 'w', encoding='utf8')
    lines = [line[0]+'\n' for line in data_list]
    vocab_file.writelines(lines)
    vocab_file.close()

    print('Training Tokenizer... \n')
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.enable_padding(length=MAX_TOKEN_LEN)
    tokenizer.enable_truncation(max_length=MAX_TOKEN_LEN)

    paths = ['helper/vocabulary.txt']
    tokenizer.train(files=paths, vocab_size=50265, min_frequency=2,
                    special_tokens=['<s>', '<pad>', '</s>', '<unk>'])

    if not os.path.exists('helper/tokenizer'):
        os.mkdir('helper/tokenizer')
    tokenizer.save_model('helper/tokenizer')