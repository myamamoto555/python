# coding:utf-8                                                                                                                           
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, Chain, optimizers, Variable, cuda
from chainer.training import extensions

import time
import os
import pickle

os.environ['PATH'] += ':/usr/local/cuda-8.0/bin:/usr/local/cuda-8.0/bin'
xp = cuda.cupy


class RNN(Chain):
    def __init__(self, src_wnum, targ_wnum):
        super(RNN, self).__init__(
            embed = L.EmbedID(src_wnum, 50, ignore_label=-1),
            enc = L.LSTM(50, 50),
            dec = L.LSTM(50, 50),
            trgembed = L.EmbedID(targ_wnum, 50, ignore_label=-1),
            he = L.Linear(50, 50),
            reembed = L.Linear(50, targ_wnum),
        )

    def __call__(self, s, t):
        loss = 0
        for s in s:
            s = Variable(cuda.to_gpu(xp.array(s, dtype = xp.int32)))
            emb_s = self.embed(s)
            hs = self.enc(emb_s)

        cell = Variable(self.xp.zeros((64, 50), dtype='float32'))
        self.dec.set_state(cell, hs)
        for i in range(len(t) - 1):
            t_in = Variable(cuda.to_gpu(xp.array(t[i], dtype = xp.int32)))
            t_out = Variable(cuda.to_gpu(xp.array(t[i+1], dtype = xp.int32)))
            emb_t = self.trgembed(t_in)
            ht = self.dec(emb_t)
            y = self.reembed(F.tanh(self.he(ht)))
            loss += F.softmax_cross_entropy(y, t_out)
        
        self.enc.reset_state()
        self.dec.reset_state()

        return loss

def train(src, targ, src_vnum, trg_vnum, epoch):
    rnn = RNN(src_vnum, trg_vnum)
    chainer.cuda.get_device("0").use()
    rnn.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(rnn)

    for epoch in range(epoch):
        all_loss = 0
        start = time.time()
        for (s, t) in zip(src, targ):
            loss = rnn(xp.array(s).T, xp.array(t).T)
            all_loss += loss.data
            rnn.zerograds()
            loss.backward()
            optimizer.update()
        end = time.time()
        print all_loss, end - start

    chainer.serializers.save_npz("model.npz", rnn)


def testcode():
    src = [[[0, 1, 2, 3, 4, -1], [0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]]
    targ = [[[4, 3, 2, 1, 0], [4, 3, 2, 1, 0]], [[4, 3, 2, 1, 0], [4, 3, 2, 1, 0]]]

    train(src, targ, 6, 5, 100)


if __name__ == '__main__':
    testcode()
