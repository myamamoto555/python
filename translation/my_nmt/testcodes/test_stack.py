import chainer.functions as F
import numpy as np


a = np.arange(6).reshape(2,3)

b = F.stack([a[i,:] for i in range(2)], axis=0)
print a
print b
