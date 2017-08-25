# coding: utf-8

import chainer.functions as F
import numpy as np


a = np.array([[1,2,3],[4,5,6]], dtype=np.float32)

b = F.expand_dims(a, 1)
print b
# >>>[[[1,2,3],[4,5,6]]]

c = F.broadcast_to(b, [2,4,3])
print c


d = F.reshape(c,[2*4,3])
print d


e = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]], dtype=np.float32)
f = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]], dtype=np.float32)

#print e.shape

#print F.concat((e,f),1)
