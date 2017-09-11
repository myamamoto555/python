import chainer
import chainer.variable as Variable
import chainer.functions as F
import numpy as np


a = np.arange(12).reshape(3,4)
b = np.arange(12).reshape(3,4)

print a
print b

a = chainer.Variable(np.array(a,np.float32))
b = chainer.Variable(np.array(b,np.float32))



print F.sum(F.batch_matmul(a, b, transa=True),1)
