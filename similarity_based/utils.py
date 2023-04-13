import math
import string
from collections import defaultdict
from nltk.corpus import stopwords
import gensim
import torch
import torch.nn.functional as F

model = gensim.models.KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True)
stop_words = set(stopwords.words("english"))


def tokenize(sentence):
    # Convert to lowercase
    sentence = sentence.lower()

    # Remove punctuation
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))

    # Tokenize the sentence and remove stopwords
    tokens = [word for word in sentence.split() if word not in stop_words]

    return tokens


def dict_compute(file):
    # Load the list of stopwords

    # Initialize a dictionary to store the word counts
    word_count_dict = defaultdict(float)
    num_words = 0
    icf_min = math.inf
    icf_max = -math.inf
    # Read the file and process it
    with open(file, "r", encoding='utf8') as file:
        for line in file:
            # Split the line into sequences and labels
            sentence1, sentence2, _ = line.strip().split("_!_")

            # Tokenize the sequences
            tokens = tokenize(sentence1) + tokenize(sentence2)
            num_words = len(tokens)

            # Update the word count dictionary
            for token in tokens:
                word_count_dict[token] += 1

    for v in word_count_dict.values():
        if v > icf_max:
            icf_max = v
        if v < icf_min:
            icf_min = v
    for k, v in word_count_dict.items():
        # icf = v / num_words
        icf = v
        word_count_dict[k] = 1 - (icf - icf_min) / (icf_max - icf_min)
    # Print the word count dictionary
    return dict(word_count_dict)


def compute_sentence(tokens_t, tokens_h, word_dict, average, threshold):
    total_sim = 0
    total_weight = 0
    for Hj in tokens_h:
        weight = 0
        if Hj in word_dict:
            weight = word_dict[Hj]
        else:
            weight = average

        max_sim = max(compute_similarity(Ti, Hj) for Ti in tokens_t)
        if max_sim == 0:
            max_sim = -1
        total_sim += max_sim * weight
        total_weight += weight
    if total_weight == 0:
        return True
    sim = total_sim / total_weight
    if sim >= threshold:
        return True
    else:
        return False


def compute_similarity(word1, word2):
    if word1 in model and word2 in model:
        vec1 = torch.FloatTensor(model[word1].copy())
        vec2 = torch.FloatTensor(model[word2].copy())
        cos_sim = F.cosine_similarity(vec1, vec2, dim=0)
        return cos_sim
    else:
        return 0.2
