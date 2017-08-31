#coding:utf-8

import chainer
import chainer.functions as F
import chainer.links as L

xp = chainer.cuda.cupy

def _mkivar(array):
    m = xp
    return chainer.Variable(m.array(array, dtype=m.int32))


class AutoEncoder(chainer.Chain):
    def __init__(self, voc_size, hidden_size):
        super(AutoEncoder, self).__init__(
            x_h = L.Linear(voc_size, hidden_size),
            h_x = L.Linear(hidden_size, voc_size),
            w_r = L.EmbedID(voc_size, hidden_size, ignore_label=-1))
        self.voc_size = voc_size

    def forward(self, x, t):
        # h's shape = (batch_size, hidden_size)
        # r's shape = (batch_size, hidden_size)
        x = xp.array(x, dtype=xp.float32)
        h = F.relu(self.x_h(x))
        o = self.h_x(h)
        #r = self.w_r(_mkivar(t))
        #e = F.matmul(h, r, transa=True)
        return F.softmax_cross_entropy(o, _mkivar(t))
