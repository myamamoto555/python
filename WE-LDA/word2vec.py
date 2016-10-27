# -*- coding: utf-8 -*-
from gensim.models import word2vec

#gensim model creation
def vecmodelcreate():
    model_path = "/home/yamamoto/project/moral/main_program/GoogleNews-vectors-negative300.bin.gz"
    model = word2vec.Word2Vec.load_word2vec_format(model_path, binary=True)

    return model


if __name__ == '__main__':
    model = vecmodelcreate()
    print model["apple"]
