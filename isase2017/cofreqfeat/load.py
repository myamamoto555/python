import numpy as np

a = list(np.load("train1_cofreq.npy"))
b = list(np.load("train2_cofreq.npy"))
c = list(np.load("train3_cofreq.npy"))

print len(a)
print len(b)
print len(c)

a.extend(b)
a.extend(c)

print len(a)


print np.load("train1_cofreq.npy")
print len(list(np.load("tdoc.npy")))
