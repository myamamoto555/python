#coding:utf-8

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.variable as Variable
import math
import numpy as np
import os

os.environ['PATH'] += ':/usr/local/cuda-8.0/bin:/usr/local/cuda-8.0/bin'
xp = chainer.cuda.cupy

def _mkivar(array):
    m = xp
    return chainer.Variable(m.array(array, dtype=m.float32))

def _mkintvar(array):
    m = xp
    return chainer.Variable(m.array(array, dtype=m.int32))


class AutoEncoder(chainer.Chain):
    def __init__(self, voc_size, topic_size, hidden_size, dec_size, dv_size):
        super(AutoEncoder, self).__init__(
            x_y = L.Linear(voc_size, hidden_size),
            y_theta = L.Linear(hidden_size, topic_size),
            theta_k = L.Linear(topic_size, hidden_size),
            R = L.Linear(hidden_size, voc_size))
        self.voc_size = voc_size
        self.hidden_size = hidden_size
        self.topic_size = topic_size
        self.dec_size = dec_size
        self.dv_size = dv_size

    def forward(self, x, trg, decade):
        #dv = self.dec_dv(_mkintvar(decade))
        y = F.relu(self.x_y(_mkivar(x)))
        #dvy = F.concat((y, dv), 1)
        #d = F.relu(self.y_d(y))
        theta = F.softmax(self.y_theta(y))
        #dvtheta = F.concat((theta, dv), 1)
        k = F.relu(self.theta_k(theta))
        m = F.clip(F.softmax(self.R(k)), 1e-10, 1.0)
        #m = F.softmax(self.R(theta))
        loss = -F.sum(F.log(m) * _mkivar(trg)) 
        return loss

    def get_theta(self, x):
        theta = self.encode(_mkivar(x))
        return theta

    def get_phi(self):
        topics = []
        for i in range(self.topic_size):
            topic = [[0.0 for _ in range(self.topic_size)]]
            topic[0][i] = 1.0
            topics.append(topic)
        theta = _mkivar(topics)
        k = F.relu(self.theta_k(theta))
        E = F.softmax(self.R(k))

        return E.data

    def show_topics(self, num, id2word):
        phi = self.get_phi()
        topic_word = {}
        for i in range(self.topic_size):
            npphi = phi[i]
            topic_word[i] = []
            for j in range(num):
                ind = np.argmax(npphi)
                topic_word[i].append(id2word[str(ind)])
                npphi[ind] = -1

        return topic_word
