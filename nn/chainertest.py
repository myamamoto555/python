# coding:utf-8

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable

import numpy as np

x1 = Variable(np.array([1,5], dtype='int32'))
x2 = Variable(np.array([1,2,3],dtype='int32'))

xs = [x1, x2]

print F.pad_sequence(xs)

x1 = Variable(np.array([[1,5],[2,3]], dtype='int32'))
x2 = Variable(np.array([[1,2,3],[7,8,9]],dtype='int32'))

xs = [x1, x2]
print x1.shape

print F.pad_sequence(xs)
