import numpy as np
from lda_sentenceLayer import lda_gibbs_sampling1

topics = 20
alpha, beta = 0.5 / float(topics), 0.5 / float(topics)
iterations = 201

def test():
    voc_num = 3
    doc1 = np.array([np.array([0, 1, 2], dtype=np.int32), np.array([0, 1],dtype=np.int32)])
    train = []
    train.append(doc1)

    train = np.array(train)

    lda = lda_gibbs_sampling1(K=topics, alpha=alpha, beta=beta, docs=train, V=voc_num)


filepath = "/home/yamamoto/project/gitrepos/python/Wiki15/wiki15prepro/"
voclist = []
train = []

for i in range(1200):
    doctmp = []
    with open(filepath + str(i)) as f:
        for l in f:
            tmp = []
            l = l.split("\n")[0]
            l = l.split("\r")[0]
            ws = l.split()
            for w in ws:
                if w not in voclist:
                    voclist.append(w)
                tmp.append(voclist.index(w))
            doctmp.append(np.array(tmp))
    train.append(np.array(doctmp))

lda = lda_gibbs_sampling1(K=topics, alpha=alpha, beta=beta, docs=train, V=len(voclist))


for i in range(iterations):
    lda.inference()
    if i%20 == 0:
        print i


ndk = lda.topicdist()
fout = open("ndk200", "w")

for i in range(ndk.shape[0]):
    for j in range(ndk[i].size):
        fout.write(str(ndk[i][j]) + " ")
    fout.write("\n")

fout.close()
