# coding:utf-8

import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np


a = np.array([[1,2,3,0], [0,0,0,0]])
t = np.array([1,2,3,4])
f = np.array([-1,-2,-3,-4])
print a.shape
for a in a:
    b = chainer.Variable(a!=0)
    print b.data
    c = F.where(b, t, f)
    print c.data


a = np.array([[1,2,3,0], [0,0,0,0]])
t = np.array([[1,2,3,4], [5,6,7,8]]) # lstmで計算されたもの
f = np.array([[0,0,0,0], [0,0,0,0]]) # 0ベクトルだった時
b = chainer.Variable(a!=0)
print b.data
c = F.where(b, t, f)
print c.data
