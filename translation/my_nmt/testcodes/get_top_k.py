# coding:utf-8

import numpy as np
from chainer import cuda

# return top-k index and scores
# input: x = np.array([[1,2,3],[4,5,6]]), k=2
# return: ([(1,2),(1,1)],[6,5])
def top_k(x, k):
    ids_list = []
    scores_list = []
    for i in range(k):
        ids = np.unravel_index(x.argmax(), x.shape)
        score = x[ids[0]][ids[1]]
        ids_list.append(ids)
        scores_list.append(score)
        x[ids[0]][ids[1]] = -1000000
    return ids_list, scores_list


x = np.array([[1,2,3],[4,5,6]])
print x
print top_k(x, 2)
print np.zeros(5)


def top_k_batch(xs, k, b):
    ids_list = []
    scores_list = []
    for b in range(b):
        x = xs[k*b:k*(b+1)]
        for i in range(k):
            ids = np.unravel_index(x.argmax(), x.shape)
            score = x[ids[0]][ids[1]]
            ids_list.append((ids[0]+k*b,ids[1]))
            scores_list.append(score)
            x[ids[0]][ids[1]] = -1000000
    return ids_list, scores_list


x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print x
print top_k_batch(x, 1, 4)
