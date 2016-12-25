# coding:utf-8
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, Chain, optimizers, Variable, cuda
from chainer.training import extensions

import time
import os

class RNN(Chain):
    def __init__(self):
        super(RNN, self).__init__(
            embed=L.EmbedID(8729, 100),
            mid=L.LSTM(100, 100),
            l = F.Linear(100, 2),
        )

    def reset_state(self):
        self.mid.reset_state()

    def __call__(self, x_list):
        for x in x_list:
            x = Variable(np.array([x], dtype = np.int32))
            emb_x = self.embed(x)
            h = self.mid(emb_x)

        return self.l(h)


for i in range(300):
    print i
    rnn = RNN()
    model = L.Classifier(rnn) # default loss function = softmax cross entropy
    chainer.serializers.load_npz("lstm100/my.model_" + str(i), model)

    #chainer.cuda.get_device("0").use()
    #model.to_gpu()

    optimizer = optimizers.SGD()
    optimizer.setup(model)

    sentences = list(np.load("sentences_test.npy"))
    evaluation = list(np.load("evaluation_test.npy"))

    for i in range(len(evaluation)):
        if float(evaluation[i]) > 3:
            evaluation[i] = 1
        else:
            evaluation[i] = 0


    correct_count = 0
    false_count = 0
    for s, e in zip(sentences, evaluation):
        rnn.reset_state()
        model.zerograds()
        y = model.predictor(s)
        correct = e
        predict = y.data.argmax()
        if correct == predict:
            correct_count += 1
        else:
            false_count += 1

    print correct_count / float(correct_count + false_count)
