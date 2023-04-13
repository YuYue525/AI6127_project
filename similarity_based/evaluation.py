'''s1 0:irrelated 1:related    s2 0:contradiction 1:entailment 2:neutral'''


def is_correct(s1, s2):
    if s1 == '0' and s2 == '2':
        return True
    if s1 == '1' and s2 == '0' or s2 == '1':
        return True
    return False


def accuracy(file_path):
    total_num = 0
    true_num = 0
    with open(file_path, 'r') as f:
        for line in f:
            total_num += 1
            if is_correct(line[0], line[1]):
                true_num += 1

    return true_num/total_num


def recall(file_path):
    total_num = 0
    true_num = 0
    with open(file_path, 'r') as f:
        for line in f:
            if line[1] in ['0', '1']:
                total_num += 1
                if line[0] == '1':
                    true_num += 1

    return true_num/total_num


def precision(file_path):
    total_num = 0
    true_num = 0
    with open(file_path, 'r') as f:
        for line in f:
            if line[0] == '1':
                total_num += 1
                if line[1] in ['0', '1']:
                    true_num += 1

    return true_num/total_num


def f1(p, r):
    return 2*p*r/(p+r)
