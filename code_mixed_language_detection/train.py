import pandas as pd
import os
import argparse
import torch
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import DataLoader
from helper.CustomDataset import CustomDataset
from helper.MixCodeModel import MixCodeModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_TOKEN_LEN = 350

def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('training_filename', help="location to training csv file", type=str)
    parser.add_argument('save_model_path', help='location to save the model', type=str)

    parser.add_argument('-e', '--epochs', help='number of epochs, default: 1', type=int, default=1)
    parser.add_argument('-m', '--load_model', help='already trained model default: None', type=str, default=None)
    parser.add_argument('-lr', '--learning_rate', help='learning rate, default: 1e-5', type=float, default=1e-5)
    parser.add_argument('-b', '--batch_size', help='trainig batch_size, default: 6', type=int, default=6)


    args = parser.parse_args()
    return args


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def train(epoch):
    if not os.path.exists('models/checkpoints/'):
        os.mkdir('models/checkpoints')
    model.train()
    for batch_count, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)
        outputs = model(ids, mask)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if batch_count%100 == 0:
            print(f'Epoch: {epoch}, batch Count: {batch_count}, Loss:  {loss.item()}')
        if batch_count%5000 == 0 and batch_count != 0:
            torch.save(model, 'models/model_checkpoint_'+str(batch_count)+'.bin')
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    args = ParseArguments()

    try:
        training_data = pd.read_csv(args.training_filename)
        print('File imported successfully...')
        if args.load_model != None:
            model = torch.load(args.load_model, map_location=device)
            model.eval()
            print('Model loaded successfully...')
        else:
            model = MixCodeModel()
            model.to(device)
            print('Model created successfully...')
    except Exception as e:
        print(e)
        exit()


    tokenizer = ByteLevelBPETokenizer('helper/tokenizer/vocab.json', 'helper/tokenizer/merges.txt')
    tokenizer.enable_padding(length=MAX_TOKEN_LEN)
    tokenizer.enable_truncation(max_length=MAX_TOKEN_LEN)

    training_set = CustomDataset(training_data, tokenizer)
    training_loader = DataLoader(training_set, batch_size=args.batch_size)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.learning_rate)

    print('\nTraining.....\n')
    print(f"Total batches: {len(training_data)/args.batch_size}\n")
    for epoch in range(args.epochs):
        train(epoch)

    torch.save(model, args.save_model_path)
