import argparse
import pandas as pd
import os
from happytransformer import HappyTextToText
import progressbar
from multiprocessing import freeze_support
from params import testing_args


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('process', help='evaluate a csv file or generate corrected sentences, eval/gen', default='generate')
    parser.add_argument('model_path', help='path to the model', type=str)
    parser.add_argument('-s', '--sentence', help='if you want to predict only one sentence, default=None', type=str, default=None)
    parser.add_argument('-f', '--file', help='if you want to multiple sentences in a file, default=None', type=str, default=None)
    args = parser.parse_args()
    return args


def prepare_data(df):
    df.columns = ['input', 'target']
    df['input'] = 'grammar: ' + df['input']
    return df


if __name__ == '__main__':
    freeze_support()
    args = parseArguments()

    model = HappyTextToText('t5', args.model_path)

    if args.process == 'eval':
        if args.file:
                testing_data = pd.read_csv(args.file)
                processed_data = prepare_data(testing_data)
                test_file = os.path.join(os.path.dirname(args.file),'prepared_'+os.path.basename(args.file))
                processed_data.to_csv(test_file, index=False)

                result = model.eval(test_file)
                print(result)
        else:
            print('CSV File is neccessary for the evaluation...')
            exit()

    if args.process == 'gen':
        if args.sentence:
            generated_text = model.generate_text('grammar: '+args.sentence, args=testing_args)
            print('\nCorrected sentence: '+generated_text.text)

        if args.file:
            read_file = open(args.file, 'r', encoding='utf8')
            write_filename = os.path.join(os.path.dirname(args.file), 'corrected_'+os.path.basename(args.file))
            write_file = open(write_filename, 'w', encoding='utf8')

            lines = read_file.readlines()
            text_list = []

            print('\nGenerating correction file...')
            bar = progressbar.ProgressBar(maxval=len(lines), widgets=[progressbar.Bar(), ' ', progressbar.SimpleProgress()])
            bar.start()
            sentence_count = 0
            for line in lines:
                line = 'grammar: ' + line.strip()
                output = model.generate_text(line, args=testing_args)
                text_list.append(line[9:] + ' --> ' +output.text.strip()+'\n')
                sentence_count += 1
                bar.update(sentence_count)
            write_file.writelines(text_list)
            bar.finish()

            read_file.close()
            write_file.close()
            print('File stored at '+ write_filename)
        
        if not args.sentence and not args.file:
            print('Either a sentence or a file is necessary...')
            exit()
