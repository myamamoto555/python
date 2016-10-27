from scipy.stats import norm
import numpy as np
print norm.pdf(x=1.0, loc=0, scale=1)

def mnd(_x, _mu, _sig):
    x = np.matrix(_x)
    mu = np.matrix(_mu)
    sig = np.matrix(_sig)
    a = np.sqrt(np.linalg.det(sig)*(2*np.pi)**sig.ndim)
    b = np.linalg.det(-0.5*(x-mu)*sig.I*(x-mu).T)
    return np.exp(b)/a


x = np.array([1, 1])
l = np.array([0.1, 1])
s = np.matrix([[1, 0], [0, 1]])
print mnd(x, l, s)

a = np.array([1,2,3])

print 5 * a
print a / float(2)

a = np.matrix([[1,0,3],[2,1,4]])
print a[:, 1]
print a[1:]


a = np.array([0,1,2])
b = np.array([3,4,5])

print a * b
