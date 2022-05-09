import os
import progressbar
import string


# f = open('data/raw/'+file_name, 'r', encoding='cp850')
# w_file = open('data/transformed/'+file_name, 'w', encoding='cp850')
# i=0
# for i, line in enumerate(f):
#       pass
# print (i + 1)

MAX_LEN = 200
TRAIN_SENTENCES_PER_LANGUAGE = 50000
TEST_SENTENCES_PER_LANGUAGE = 10000


raw_dir = 'data/1.raw/'
train_filter_dir = 'data/2.filtered/'
test_filter_dir = 'data/2.test_filtered/'


if not os.path.exists(train_filter_dir):
    os.mkdir(train_filter_dir)
if not os.path.exists(test_filter_dir):
    os.mkdir(test_filter_dir)

data_files = os.listdir(raw_dir)

for data_file in data_files:
    f = open(raw_dir+data_file, 'r', encoding='utf8')
    train_w_file = open(train_filter_dir+data_file, 'w', encoding='utf8')
    test_w_file = open(test_filter_dir+data_file, 'w', encoding='utf8')

    sentence_count = 0
    print("Pre-filtering: "+data_file)
    bar = progressbar.ProgressBar(maxval=TRAIN_SENTENCES_PER_LANGUAGE, widgets=[progressbar.Bar(), ' ', progressbar.SimpleProgress()])
    bar.start()

    while sentence_count < TRAIN_SENTENCES_PER_LANGUAGE:
        line = f.readline()
        line = line.translate(str.maketrans('', '', string.punctuation)).strip()
        words = line.split(' ')
        if len(words) >= 2 and len(words) <= MAX_LEN and not line.strip().isnumeric():
            sentence_count += 1
            bar.update(sentence_count)
            train_w_file.write(line.strip() + '\n')
    bar.finish()

    sentence_count = 0
    print('Test pre filtering')
    bar = progressbar.ProgressBar(maxval=TEST_SENTENCES_PER_LANGUAGE, widgets=[progressbar.Bar(), ' ', progressbar.SimpleProgress()])
    bar.start()
    while sentence_count < TEST_SENTENCES_PER_LANGUAGE:
        line = f.readline()
        line = line.translate(str.maketrans('', '', string.punctuation)).strip()
        words = line.split(' ')

        if len(words) >= 2 and len(words) <= MAX_LEN and not line.strip().isnumeric():
            sentence_count += 1
            bar.update(sentence_count)
            test_w_file.write(line.strip() + '\n')
    bar.finish()

f.close()
train_w_file.close()
test_w_file.close()
