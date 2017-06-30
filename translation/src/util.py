#coding:utf-8
import numpy as np

def pading(batch_array):
    conv_batch_array = []
    for ba in batch_array:
        conv_batch_array.append(list(ba))
    for ba in conv_batch_array:
        max_len = 0
        for b in ba:
            if len(b) > max_len:
                max_len = len(b)
        for b in ba:
            for i in range(len(b), max_len):
                b.append(-1)
    return conv_batch_array


def batch_create(src, trg, batchsize):
    N = len(src)
    perm = np.random.permutation(N)
    src_batch = []
    trg_batch = []
    for i in range(0, N, batchsize):
        src_batch.append(src[perm[i:i + batchsize]])
        trg_batch.append(trg[perm[i:i + batchsize]])

    srb = pading(src_batch)
    trb = pading(trg_batch)

    return srb, trb


def test():
    src = np.array([[0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]])
    trg = np.array([[0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]])

    srb, trb = batch_create(src, trg, 2)
    for sr in srb:
        sr = np.array(sr)
        print sr.shape
        


if __name__ == '__main__':
    test()
