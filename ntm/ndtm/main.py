#coding:utf-8

import chainer
import chainer.functions as F
import util
import random
import model
import numpy as np

in_file = './nips/nips_lem_cut.txt'
voc_file = './nips/voc.txt'
voc_size = 8643
wid_file = './nips/wid.txt'
xp = chainer.cuda.cupy
topic_size = 50
hidden_size = 256
dec_size = 29
dv_size = 15
gpu = 0

def create_batch(x):
    random.shuffle(x)
    ret = []
    for i, x in enumerate(x):
        if i%64 == 0:
            if i != 0:
                ret.append((in_tmp, out_tmp, times))
            in_tmp = []
            out_tmp = []
            times = []
        in_tmp.append(x[0])
        out_tmp.append(x[1])
        times.append(x[2])

    return ret


if __name__ == '__main__':
    # create vocabulary files.                                                          
    util.make_vocabs(in_file, voc_file, voc_size)

    # convert word to word_id
    id2word = util.apply_vocabs(in_file, wid_file, voc_file)

    datas = []
    c = 0
    with open(wid_file) as f:
        for l in f:
            l = l.split("\n")[0]
            time = int(l.split("\t")[0])-1987
            l = l.split("\t")[1]
            xs = l.split()
            bowfreq = [0 for _ in range(voc_size)]
            for x in xs:
                bowfreq[int(x)] += 1
                c += 1
            datas.append((bowfreq, bowfreq, time))
    print c
    
    # model setup                                                                                                      
    mdl = model.AutoEncoder(voc_size, topic_size, hidden_size, dec_size, dv_size)
    # optimizer set up                                                                                                 
    opt = chainer.optimizers.Adam()
    opt.setup(mdl)
    # use gpu                                                                                                          
    chainer.cuda.get_device(gpu).use()
    mdl.to_gpu()

    for i in range(1000):
        all_loss = 0
        batches = create_batch(datas)
        for b in batches:
            mdl.zerograds()
            loss = mdl.forward(b[0], b[1], b[2])
            loss.backward()
            opt.update()
            all_loss += loss.data
        print all_loss

    topic_word = mdl.show_topics(10, id2word)
    for k, v in topic_word.items():
        print k, v