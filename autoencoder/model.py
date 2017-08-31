#coding:utf-8

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.variable as Variable

xp = chainer.cuda.cupy

def _mkivar(array):
    m = xp
    return chainer.Variable(m.array(array, dtype=m.int32))


class AutoEncoder(chainer.Chain):
    def __init__(self, voc_size, topic_size, hidden_size):
        super(AutoEncoder, self).__init__(
            x_h = L.Linear(voc_size, topic_size),
            wvec = L.Linear(hidden_size, voc_size),
            tvec = L.Linear(hidden_size, topic_size))
        self.voc_size = voc_size
        self.hidden_size = hidden_size

    def forward(self, x, t):
        x = xp.array(x, dtype=xp.float32)
        h = F.softmax(self.x_h(x))
        I = xp.identity(self.hidden_size, dtype=xp.float32)
        w = self.wvec(I)
        beta = F.softmax(self.tvec(F.transpose(w)), 0)
        predict = F.matmul(h, beta, transb=True)
        return F.softmax_cross_entropy(predict, _mkivar(t))
