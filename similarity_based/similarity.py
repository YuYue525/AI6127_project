from utils import *
from evaluation import *

word_dict = dict_compute('./data/train.txt')
total = 0
for v in word_dict.values():
    total += v

average = total / len(word_dict.values())

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for th in thresholds:
    with open('./data/test.txt', "r", encoding='utf8') as file:
        with open('./test_result/test_data_{}.txt'.format(th), 'w') as out:
            for line in file:
                # Split the line into sequences and labels
                sentence1, sentence2, label = line.strip().split("_!_")
                tokens_t = tokenize(sentence1)
                tokens_h = tokenize(sentence2)
                if len(tokens_t) > 4 and len(tokens_h) > 4:
                    out.write(str(int(compute_sentence(
                        tokens_t, tokens_h, word_dict, average, th))) + str(label) + '\n')

    print('---- threshold: {}   finished'.format(th))


with open('result.txt','w') as result:
    for th in thresholds:
        print('- threshold: {} evaluating'.format(th))
        file_path = './test_result/test_data_{}.txt'.format(th)

        r = recall(file_path)
        p = precision(file_path)
        result.write('-- threshold: {}'.format(th))
        result.write('-- accuracy = {}'.format(accuracy(file_path)))
        result.write('-- f1_score = {}'.format(f1(p,r)))
        result.write('-- precision = {}'.format(p))
        result.write('-- recall = {}'.format(r))
        result.write('\n')
