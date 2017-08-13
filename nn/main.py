#coding:utf-8

from chain import *
import numpy as np


if __name__ == '__main__':
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    t = np.array([[1], [0], [0], [1]], dtype=np.float32)

    model = Chain(
        l1 = Linear(2, 5),
        l2 = Linear(5, 1)
    )
    
    def forward(x):
        h = model.l1(x)
        h = relu(h)
        h = model.l2(h)
        return h

    opt = SGD(lr=0.1)
    opt.setup(model)

    x = Variable(x)
    t = Variable(t)

    for epoch in range(1000):
        y = forward(x)
        loss = mean_squared_error(y, t)
        model.zerograds()
        loss.backward()
        opt.update()
        print loss.data
    
        
    print forward(x).data
