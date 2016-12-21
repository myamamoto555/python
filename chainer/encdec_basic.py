# coding:utf-8

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, Chain, optimizers, Variable
from chainer.training import extensions


class RNN(Chain):
    def __init__(self):
        super(RNN, self).__init__(
            embed=L.EmbedID(1000, 100),  # word embedding
            mid=L.LSTM(100, 50),  # the first LSTM layer
            out=L.Linear(50, 1000),  # the feed-forward output layer
        )

    def reset_state(self):
        self.mid.reset_state()

    def __call__(self, cur_word):
        # Given the current word ID, predict the next word.
        x = self.embed(cur_word)
        h = self.mid(x)
        y = self.out(h)
        return y


def compute_loss(x_list):
    loss = 0
    for cur_word, next_word in zip(x_list, x_list[1:]):
        cur_word = Variable(np.array([cur_word], dtype = np.int32))
        next_word = Variable(np.array([next_word], dtype = np.int32))
        loss += model(cur_word, next_word)
    return loss


rnn = RNN()
model = L.Classifier(rnn)
optimizer = optimizers.SGD()
optimizer.setup(model)

x_list = [1,2,3,4,5,6]

for i in range(50):
    rnn.reset_state()
    model.zerograds()
    loss = compute_loss(x_list)
    loss.backward()
    optimizer.update()
    print loss.data
