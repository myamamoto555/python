#coding:utf-8
from collections import Counter
import numpy as np

dim = 30000 #dimension of feature

#入力：リスト　出力：頻度上位dim件の単語
def count(data):
    features = []
    counter = Counter(data)
    for word, cnt in counter.most_common(dim):
        features.append(word)
    return features

def ngram(input, n):
    last = len(input) - n + 1
    ret = []
    for i in range(0, last):
        ret.append("-".join(input[i:i+n]))
    return ret

def get_feature(sentences):
    feature_list = []
    for ws in sentences:
        feature_list.extend(ws)
        feature_list.extend(ngram(ws, 2))
        feature_list.extend(ngram(ws, 3))
    features = count(feature_list)
    
    final = []
    for ws in sentences:
        sent_feature = np.zeros(len(features))
        sflist = []
        sflist.extend(ws)
        sflist.extend(ngram(ws, 2))
        sflist.extend(ngram(ws, 3))
        for sf in sflist:
            if sf in features:
                index = features.index(sf)
                sent_feature[index] = 1
        final.append(sent_feature)
    return features, final

if __name__ == '__main__':
    sentences = [["a", "b", "c"], ["a", "a", "b", "b"]]
    print get_feature(sentences)

