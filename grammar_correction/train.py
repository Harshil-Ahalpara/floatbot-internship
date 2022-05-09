import argparse
import pandas as pd
import os
from happytransformer import HappyTextToText
from multiprocessing import freeze_support
from params import training_args


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('training_filename', help='path to training csv file containing "input" and "target" columns', type=str)
    parser.add_argument('save_model_path', help='path to save the model', type=str)
    parser.add_argument('-m', '--model', help='previously trained model for further training, default=None', type=str, default=None)
    args = parser.parse_args()
    return args


def prepare_data(df):
    df.columns = ['input', 'target']
    df['input'] = 'grammar: ' + df['input']
    return df


if __name__ == '__main__':
    freeze_support()
    args = parseArguments()

    transformer_model = 't5-base'

    try:
        training_data = pd.read_csv(args.training_filename)
        print('File imported successfully...')

        if args.model:
            model = HappyTextToText('t5', args.model)
            print('Model loaded successfully...')
        else:
            print(f'Using {transformer_model} model...')
            # Checking if the local copy of t5-base exists 
            if not os.path.exists(f'models/{transformer_model}-local'):
                if not os.path.exists('models'):
                    os.mkdir('models')
                model = HappyTextToText('t5', transformer_model)
                model.save(f'models/{transformer_model}-local')
            else:
                print('Using localy stored model')
                model = HappyTextToText('t5', f'models/{transformer_model}-local')
            print('Model created successfully...')

    except Exception as e:
        print(e)
        exit()

    training_data = pd.read_csv('data/train_data.csv')
    processed_data = prepare_data(training_data)
    prepared_file_name = os.path.join(os.path.dirname(args.training_filename),'prepared_'+os.path.basename(args.training_filename))
    processed_data.sample(frac=1)
    processed_data.to_csv(prepared_file_name, index=False)

    model.train(prepared_file_name, args=training_args)
    model.save(args.save_model_path)

    print("\n\n Model saved at: ", args.save_model_path)
