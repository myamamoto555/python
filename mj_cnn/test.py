# coding:utf-8

import chainer
import chainer.functions as F
import numpy as np

if __name__ == '__main__':
    z = chainer.Variable(np.array([[1, 0.5],[3,2]], dtype=np.float32))
    t = chainer.Variable(np.array([1,0], dtype=np.int32))
    print F.accuracy(z, t)
