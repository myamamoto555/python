# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L
import numpy
import numpy as np
import os
import copy

os.environ['PATH'] += ':/usr/local/cuda-8.0/bin:/usr/local/cuda-8.0/bin'
use_gpu = True
xp = chainer.cuda.cupy


def _array_module():
    return chainer.cuda.cupy if use_gpu else numpy


def _mkivar(array):
    m = _array_module()
    return chainer.Variable(m.array(array, dtype=m.int32))


def _zeros(shape):
    m = _array_module()
    return chainer.Variable(m.zeros(shape, dtype=m.float32))


class cnn(chainer.Chain):
    def __init__(
            self,
            vocab_size,
            embed_size):
        super(cnn, self).__init__(
            x_i = L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            i_c = L.Convolution2D(1, 128, (3, embed_size)), # in_channel, out_channel, kernel_size
            h_h = L.Linear(128, 128),
            h_z = L.Linear(128, 2))

        self.vocab_size = vocab_size
        self.embed_size = embed_size

    def forward(self, x_list):
        height = len(x_list[0])
        batch_size = len(x_list)
        xs = np.array(x_list).flatten()
        itmp = self.x_i(_mkivar(xs))
        i = F.reshape(itmp, (batch_size, 1, height, self.embed_size))
        c = self.i_c(i)
        pc = F.max_pooling_2d(c, (height, 1))
        h = F.dropout(F.reshape(pc, (batch_size, 128)), 0.2)
        h = F.dropout(F.tanh(self.h_h(h)), 0.2)
        z = self.h_z(h)
        return z

    def _loss(self, z, t):
        return F.softmax_cross_entropy(z, _mkivar(t))

    def train(self, x_list, t_list):
        z = self.forward(x_list)
        loss = self._loss(z, t_list)
        return loss

    def test(self, x_list, t_list):
        z = self.forward(x_list)
        batch_size = len(x_list)
        acc = F.accuracy(z, _mkivar(t_list)).data
        return acc * batch_size
