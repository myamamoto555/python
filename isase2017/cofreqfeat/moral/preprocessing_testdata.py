#coding:utf-8

import MeCab
import os
import numpy as np

ok_wordlist = []

def get_wakachi(text):
    tagger = MeCab.Tagger(' -Owakati')
    res = tagger.parse(text)
    words = res.split()

    return words


def fileread(filename):
    sentences = []
    evaluation = []
    with open(filename) as f:
        for l in f:
            l = l.split("\n")[0]
            s = l.split("\t")[0]
            e = l.split("\t")[1]

            sentences.append(get_wakachi(s))
            evaluation.append(e)

    return sentences, evaluation


def preprocess():
    sentences, evaluation = fileread("testset")

    with open("wordlist") as f:
        for l in f:
            w = l.split("\n")[0]
            ok_wordlist.append(w)

    unk_index = len(ok_wordlist)
    sentences_index = []
    for s in sentences:
        sentences_index_tmp = []
        for w in s:
            if w in ok_wordlist:
                sentences_index_tmp.append(ok_wordlist.index(w))
            else:
                sentences_index_tmp.append(unk_index)
        sentences_index.append(sentences_index_tmp)
    
    sentences_index = np.array(sentences_index)
    evaluation = np.array(evaluation)

    np.save("sentences_test.npy", sentences_index)
    np.save("evaluation_test.npy", evaluation)


if __name__ == '__main__':
    preprocess()
