import numpy as np
import math

print math.exp(-10000) * 10000


print (math.gamma(6) * math.gamma(3)) / math.gamma(5)
print math.exp(math.lgamma(6) + math.lgamma(3) -  math.lgamma(5))

print math.exp(math.lgamma(4)/ math.lgamma(6))
print math.gamma(4) / math.gamma(6)
print math.gamma(4)
print math.exp(math.lgamma(4))

a = np.array([1.0,1.0])
b = np.array([5, 6])
print a/b


a = np.zeros((2,3))
print a


values = [1,2,3,4,5]
print reduce(lambda x,y:x*y,values)

for i,v in enumerate(values):
    print i
