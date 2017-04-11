#coding:utf-8

import MeCab
import os
import numpy as np

wordcount = {}
ok_wordlist = []

def get_wakachi(text):
    tagger = MeCab.Tagger(' -Owakati')
    res = tagger.parse(text)
    words = res.split()
    for w in words:
        if w in wordcount:
            wordcount[w] += 1
        else:
            wordcount[w] = 1
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
    for k, v in wordcount.items():
        if v > 4:
            ok_wordlist.append(k)
    return sentences, evaluation


def preprocess():
    sentences, evaluation = fileread("alldatas")

    fout = open("wordlist", "w")
    for w in ok_wordlist:
        fout.write(w + "\n")
        
    fout.close()

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

    np.save("sentences_12.npy", sentences_index)
    np.save("evaluation_12.npy", evaluation)


if __name__ == '__main__':
    preprocess()