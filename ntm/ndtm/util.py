#coding: utf-8

from collections import defaultdict
import os

def make_vocabs(in_file, voc_file, voc_size):
    make_vocab(in_file, voc_file, voc_size)


def make_vocab(in_fp, out_fp, size):
    freq = defaultdict(int)
    num_lines = 0
    with open(in_fp) as fp:
        for line in fp:
            tmp = []
            line = line.split("\n")[0]
            line = line.split("\t")[1]
            num_lines += 1
            for word in line.split():
                freq[word] += 1

    freq_sorted = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    with open(out_fp, 'w') as fp:
        for i, (key, val) in zip(range(0, size), freq_sorted):
            fp.write('%d\t%s\t%d\n' % (i, key, val))

def apply_vocabs(in_fp, out_fp, vocab_path):
    vocab = defaultdict(lambda: '0')
    id2word = {}
    with open(vocab_path) as vocab_file:
        for line in vocab_file:
            word_id, word, freq = line.split()
            vocab[word] = word_id
            id2word[word_id] = word
    with open(in_fp) as input_file, open(out_fp, 'w') as output_file:
        for line in input_file:
            line = line.split("\n")[0]
            time = line.split("\t")[0]
            line = line.split("\t")[1]
            word_ids = [vocab[w] for w in line.split()]
            output_file.write(time + "\t")
            output_file.write(' '.join(word_ids))
            output_file.write("\n")

    return id2word
