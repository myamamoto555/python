# coding:utf-8
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, Chain, optimizers, Variable, cuda
from chainer.training import extensions

import time

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
            x = Variable(cuda.to_gpu(np.array(x, dtype = np.int32)))
            emb_x = self.embed(x)
            h = self.mid(emb_x)

        return self.l(h)


rnn = RNN()
model = L.Classifier(rnn) # default loss function = softmax cross entropy
# chainer.serializers.load_npz("my.model_200", model)

chainer.cuda.get_device("0").use()
model.to_gpu()

optimizer = optimizers.SGD()
optimizer.setup(model)

sentences = list(np.load("sentences_12.npy"))
evaluation = list(np.load("evaluation_12.npy"))

for i in range(len(evaluation)):
    if evaluation[i] == "-1":
        evaluation[i] = 0

# 長さ順にminibatchを作成
length = {}

for i, s in enumerate(sentences):
    length[i] = len(s)

sen = []
ev = []
count = 0
sen_valid = []
ev_valid = []
all_count = 0

for k, v in sorted(length.items(), key=lambda x:x[1], reverse = True):
    if v == 0:
        break

    if count % 64 == 0:
        maxlen = len(sentences[k])

    anaume = maxlen - v
    for i in range(anaume):
        sentences[k].append(8728)

    sen.append(sentences[k])
    ev.append(evaluation[k])    
    count += 1


batchnum = len(evaluation) / 64


for i in range(300):
    loss = 0

    start = time.time()
    for j in range(batchnum):
        rnn.reset_state()
        model.zerograds()
        
        s = np.array(sen[j * 64:(j + 1) * 64]).T
        t = Variable(cuda.to_gpu(np.array(ev[j * 64:(j + 1) * 64], dtype = np.int32)))

        optimizer.update(model, s, t)
        
        loss += model.loss.data

    print i, loss
    end = time.time()
    print end - start

    chainer.serializers.save_npz('lstm100/my.model_' + str(i), model)
