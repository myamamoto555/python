#coding:utf-8

import numpy as np

class Variable(object):
    def __init__(self, data, grad=None, name=None):
        self.data =data
        self.creator = None
        self.grad = grad
        self.name = name

    def set_creator(self, gen_func):
        self.creator = gen_func

    def backward(self):
        if self.creator is None:
            return
        func = self.creator
        while func:
            gy = func.output.grad
            func.input.grad = func.backward(gy)
            func = func.input.creator

    def zerograd(self):
        self.grad.fill(0)


class Function(object):
    def __call__(self, in_var):
        in_data = in_var.data
        output = self.forward(in_data)
        ret = Variable(output)
        ret.set_creator(self)
        self.input = in_data
        self.output = ret
        return ret

    def forward(self, in_data):
        NotImplementedError()

    def backward(self, grad_output):
        NotImplementedError()
        

class Link(object):
    def __init__(self, **params):
        self.dic = {}
        for name, value in params.items():
            grad = np.full_like(value, 0)
            var = Variable(value, grad, name)
            self.dic[name] = var

    def params(self):
        for param in self.__dict__.values():
            yield param

    def namedparams(self):
        for name, param in self.__dict__.items():
            yield '/' + name, param

    def zerograds(self):
        for param in self.params():
            param.zerograd()


class Chain(Link):
    def __init__(self, **links):
        super(Chaine, self).__init__()
        self.children = []
        self.dic = {}
        for name, link in links.items():
            self.children.append(name)
            self.dic[name] = link


class Linear(Link):
    def __init__(self, in_size, out_size):
        n = np.random.normal
        scale = np.sqrt(2. / in_size)
        W = n(loc=0.0, scale=scale, size=(out_size, in_size))
        b = n(loc=0.0, scale=scale, size=(out_size,))
        super(Linear, self).__init__(
            W=W.astype(np.float32), b=b.astype(np.float32))

    def __call__(self, x):
        return LinearFunction()(x, self.W, self.b)


class LinearFunction(Function):




class Optimizer(object):
    def setup(self, link):
        self.target = link
        self.status = {}
        self.prepare()

    def prepare(self):
        for name, param in self.target.namedparams():
            if name not in self.status:
                self.status[name] = {}

    def update(self):
        self.prepare()
        for name, param in self.target.namedparams():
            self.update_one(param, self.states[name])

    def update_one(self, param, state):
        NotImplementedError()


class SGD(Optimizer):
    def __init__(self, lr=0,01):
        self.lr = lr

    def update_one(self, param, state):
        param.data -= self.lr * param.grad
