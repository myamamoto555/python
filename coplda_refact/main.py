import numpy as np
from lda_sentenceLayer import lda_gibbs_sampling1

topics = 20
alpha, beta = 0.5 / float(topics), 0.5 / float(topics)
iterations = 1

voc_num = 3
doc1 = np.array([np.array([0, 1, 2], dtype=np.int32), np.array([0, 1],dtype=np.int32)])
train = []
train.append(doc1)

train = np.array(train)

lda = lda_gibbs_sampling1(K=topics, alpha=alpha, beta=beta, docs=train, V=voc_num)

for i in range(iterations):
    lda.inference()


ndk = lda.topicdist()
fout = open("ndk200", "w")

for i in range(ndk.shape[0]):
    for j in range(ndk[i].size):
        fout.write(str(ndk[i][j]) + " ")
    fout.write("\n")

fout.close()
