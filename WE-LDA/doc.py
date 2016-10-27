# coding:utf-8

import word2vec
import numpy as np

class data:
    def __init__(self, basic_file_name):
        self.vecmodel = word2vec.vecmodelcreate()
        self.vocs = []
        self.docs = []
        self.vectors = []
        with open(basic_file_name+".voc") as f:
            for l in f:
                w = l.split("\n")[0]
                self.vocs.append(w)
                self.vectors.append(self.vecmodel[w])
        with open(basic_file_name+".doc") as f:
            for l in f:
                l = l.split("\n")[0]
                ws = l.split()

                self.docs.append(ws)
    
    def average_vector(self):
        sum = np.zeros(300)
        for v in self.vectors:
            sum += v
        num = np.array([len(self.vectors) for i in range(300)])
        ave = sum / num
        return ave

if __name__ == '__main__':
    d = data("./testdocs/data")
    sum = np.zeros(300)
    for v in d.vectors:
        sum += v
    num = np.array([len(d.vectors) for i in range(300)])
    ave = sum / num
    print ave
