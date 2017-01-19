# coding:utf-8
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, Chain, optimizers, Variable, cuda
from chainer.training import extensions


class RNN(Chain):
    def __init__(self):
        super(RNN, self).__init__(
            embed=L.EmbedID(1000, 100),
            mid=L.LSTM(100, 100),
            l = F.Linear(100, 2),
        )

    def reset_state(self):
        self.mid.reset_state()

    def __call__(self, x_list):
        for x in x_list:
            x = Variable(cuda.to_gpu(np.array([x], dtype = np.int32)))
            emb_x = self.embed(x)
            h = self.mid(emb_x)
        return self.l(h)


rnn = RNN()
model = L.Classifier(rnn) # default loss function = softmax cross entropy

chainer.cuda.get_device("0").use()
model.to_gpu()

optimizer = optimizers.SGD()
optimizer.setup(model)

x_list = [1,2,3,4,5,6]
outputs = np.array([[0]], dtype=np.int32)


for i in range(50):
    rnn.reset_state()
    model.zerograds()
    t = Variable(cuda.to_gpu(outputs[0].astype(np.int32)))
    optimizer.update(model, x_list, t)

    loss = model.loss.data

    print loss

rnn.reset_state()
y = F.softmax(model.predictor(x_list))
print y.data
