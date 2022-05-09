from indictrans import Transliterator
import os
import progressbar

train_raw_dir = 'data/2.filtered/'
test_raw_dir = 'data/2.test_filtered/'
train_transformed_dir = 'data/3.transformed/'
test_transformed_dir = 'data/3.test_transformed/'

# For training set

if not os.path.exists(train_transformed_dir):
    os.mkdir(train_transformed_dir)
raw_files = os.listdir(train_raw_dir)

for raw_file in raw_files:
    if raw_file == 'bengali.txt':
        trn = Transliterator(source='ben', target='eng', build_lookup=True)
    elif raw_file == 'gujarati.txt':
        trn = Transliterator(source='guj', target='eng', build_lookup=True)
    elif raw_file == 'hindi.txt':
        trn = Transliterator(source='hin', target='eng', build_lookup=True)
    elif raw_file == 'malyalam.txt':
        trn = Transliterator(source='mal', target='eng', build_lookup=True)
    elif raw_file == 'marathi.txt':
        trn = Transliterator(source='mar', target='eng', build_lookup=True)
    elif raw_file == 'nepali.txt':
        trn = Transliterator(source='nep', target='eng', build_lookup=True)
    elif raw_file == 'punjabi.txt':
        trn = Transliterator(source='pan', target='eng', build_lookup=True)
    elif raw_file == 'tamil.txt':
        trn = Transliterator(source='tam', target='eng', build_lookup=True)
    elif raw_file == 'telugu.txt':
        trn = Transliterator(source='tel', target='eng', build_lookup=True)
    elif raw_file == 'english.txt':
        r_file = open(train_raw_dir+raw_file, 'r', encoding='utf8')
        print("Transliterating: "+raw_file)
        lines = r_file.readlines()
        w_file = open(train_transformed_dir+raw_file, 'w', encoding='utf8')
        w_file.writelines(lines)
        r_file.close()
        w_file.close()
        print("Transliterated: "+raw_file)
        continue


    in_file = open(train_raw_dir+raw_file, 'r', encoding='utf8')
    out_file = open(train_transformed_dir+raw_file, 'w', encoding='utf8')

    lines = in_file.readlines()
    count = 0

    print("Transliterating: "+raw_file)
    bar = progressbar.ProgressBar(maxval=len(lines), widgets=[progressbar.Bar(), ' ', progressbar.SimpleProgress()])
    bar.start()

    for line in lines:
        tr = trn.transform(line.strip())
        count += 1
        bar.update(count)
        out_file.write(tr+'\n')
    bar.finish()
    in_file.close()
    out_file.close()


# For test set

if not os.path.exists(test_transformed_dir):
    os.mkdir(test_transformed_dir)
raw_files = os.listdir(test_raw_dir)

for raw_file in raw_files:
    if raw_file == 'bengali.txt':
        trn = Transliterator(source='ben', target='eng', build_lookup=True)
    elif raw_file == 'gujarati.txt':
        trn = Transliterator(source='guj', target='eng', build_lookup=True)
    elif raw_file == 'hindi.txt':
        trn = Transliterator(source='hin', target='eng', build_lookup=True)
    elif raw_file == 'malyalam.txt':
        trn = Transliterator(source='mal', target='eng', build_lookup=True)
    elif raw_file == 'marathi.txt':
        trn = Transliterator(source='mar', target='eng', build_lookup=True)
    elif raw_file == 'nepali.txt':
        trn = Transliterator(source='nep', target='eng', build_lookup=True)
    elif raw_file == 'punjabi.txt':
        trn = Transliterator(source='pan', target='eng', build_lookup=True)
    elif raw_file == 'tamil.txt':
        trn = Transliterator(source='tam', target='eng', build_lookup=True)
    elif raw_file == 'telugu.txt':
        trn = Transliterator(source='tel', target='eng', build_lookup=True)
    elif raw_file == 'english.txt':
        r_file = open(test_raw_dir+raw_file, 'r', encoding='utf8')
        print("Transliterating: "+raw_file)
        lines = r_file.readlines()
        w_file = open(test_transformed_dir+raw_file, 'w', encoding='utf8')
        w_file.writelines(lines)
        r_file.close()
        w_file.close()
        print("Transliterated: "+raw_file)
        continue


    in_file = open(test_raw_dir+raw_file, 'r', encoding='utf8')
    out_file = open(test_transformed_dir+raw_file, 'w', encoding='utf8')

    lines = in_file.readlines()
    count = 0

    print("Transliterating: "+raw_file)
    bar = progressbar.ProgressBar(maxval=len(lines), widgets=[progressbar.Bar(), ' ', progressbar.SimpleProgress()])
    bar.start()

    for line in lines:
        tr = trn.transform(line.strip())
        count += 1
        bar.update(count)
        out_file.write(tr+'\n')
    bar.finish()
    in_file.close()
    out_file.close()