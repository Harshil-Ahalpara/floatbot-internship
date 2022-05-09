import sys
from simpletransformers.t5 import T5Model
import torch
import os

model_path = sys.argv[1]
inp_file_path = sys.argv[2]
save_file_path = sys.argv[3]


def get_sentences():
    #Transform  "sentence" => "paraphrase: sentence"
    with open(inp_file_path, 'r') as f:
        sentences = f.readlines()
        f.close()

    l = len(sentences)
    for i in range(l):
        sentences[i] = 'paraphrase: ' + sentences[i].strip()
    return sentences


if __name__ == '__main__':

    args = {
    "overwrite_output_dir": False,
    "max_seq_length": 256,
    "max_length": 50,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 5
    }
    # Loading previously finetuned model
    trained_model = T5Model("t5",model_path,args=args, use_cuda=torch.cuda.is_available())

    sentences = get_sentences()
    preds = trained_model.predict(sentences)

    res_file = open(save_file_path, 'a')
    l = len(sentences)
    for i in range(l):
        text = f'''
Sentence: {sentences[i][12:]}
Paraphrase:
1. {preds[i][0]}
2. {preds[i][1]}
3. {preds[i][2]}
4. {preds[i][3]}
5. {preds[i][4]}\n\n'''

        res_file.write(text)

    res_file.close()

    print("\n\nResult saved at :", save_file_path)