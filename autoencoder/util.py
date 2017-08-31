#coding: utf-8

from collections import defaultdict
import os

def create_dirs(*args):
    for dir_name in args:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

def make_vocabs(in_file, voc_file, voc_size):
    make_vocab(in_file, voc_file, voc_size)


def make_vocab(in_fp, out_fp, size):
    freq = defaultdict(int)
    num_lines = 0
    with open(in_fp) as fp:
        for line in fp:
            num_lines += 1
            for word in line.split():
                freq[word] += 1

    freq_sorted = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    with open(out_fp, 'w') as fp:
        for i, (key, val) in zip(range(0, size), freq_sorted):
            fp.write('%d\t%s\t%d\n' % (i, key, val))


def apply_vocabs(in_fp, out_fp, vocab_path):
    vocab = defaultdict(lambda: '0')
    with open(vocab_path) as vocab_file:
        for line in vocab_file:
            word_id, word, freq = line.split()
            vocab[word] = word_id
    with open(in_fp) as input_file, open(out_fp, 'w') as output_file:
        for line in input_file:
            word_ids = [vocab[w] for w in line.split()]
            output_file.write(' '.join(word_ids))
            output_file.write("\n")


def restore_words(input_file, output_file, vocab_file):
    vocab = defaultdict(lambda: '<unk>')  # unknown ID is converted into '<unk>'
    with open(vocab_file) as vocab_file:
        for line in vocab_file:
            word_id, word, freq = line.split()
            vocab[word_id] = word
    with open(input_file) as input_file, open(output_file, 'w') as output_file:
        for line in input_file:
            words = [vocab[w] for w in line.split()]
            output_file.write(' '.join(words))
            output_file.write("\n")
