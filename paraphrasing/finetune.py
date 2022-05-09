import pandas as pd
from simpletransformers.t5 import T5Model
import os
import sys
from sklearn.model_selection import train_test_split
import torch

data_dir = sys.argv[1]
trained_model_path = sys.argv[2]
save_model_path = sys.argv[3]


# Concating data from different files into single variable 
def get_prepared_data():
    files = os.listdir(data_dir)
    dataset_df = pd.DataFrame()
    for f in files:
        d = pd.read_csv(os.path.join(data_dir, f))
        dataset_df = pd.concat([dataset_df, d])
    dataset_df.columns = ["input_text","target_text"]
    dataset_df["prefix"] = "paraphrase"

    train_data,test_data = train_test_split(dataset_df,test_size=0.20)

    return train_data, test_data

if __name__ == "__main__":

    training_args = {
    'output_dir': save_model_path,
    'no_cache':True,
    "reprocess_input_data": True,
    "overwrite_output_dir": False,
    "max_seq_length": 256,
    "num_train_epochs": 1,
    "num_beams": None,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "use_multiprocessing": False,
    "save_steps": -1,
    "save_eval_checkpoints": True,
    "evaluate_during_training": False,
    "adam_epsilon": 1e-08,
    "eval_batch_size": 6,
    "fp_16": False,
    "gradient_accumulation_steps": 16,
    "learning_rate": 0.0003,
    "max_grad_norm": 1.0,
    "n_gpu": 1,
    "seed": 42,
    "train_batch_size": 6,
    "warmup_steps": 0,
    "weight_decay": 0.0,
    }

    if trained_model_path == 'None':
        model = T5Model('t5', 't5-small', args=training_args, use_cuda=torch.cuda.is_available())
        
    else:
        model = T5Model('t5', trained_model_path, args=training_args, use_cuda=torch.cuda.is_available())
    
    train_data, test_data = get_prepared_data()

    model.train_model(train_data, use_cuda=torch.cuda.is_available())

    print("\n\n Model saved at: ", save_model_path)
