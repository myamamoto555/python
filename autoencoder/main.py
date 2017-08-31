#coding:utf-8

import chainer
import chainer.functions as F
import util
import random
import model

in_file = './data/parsed_wiki15.txt'
voc_file = './data/voc.txt'
voc_size = 13895
wid_file = './data/wid.txt'
xp = chainer.cuda.cupy
hidden_size = 20
gpu = 0

def create_batch(x):
    random.shuffle(x)
    ret = []
    for i, x in enumerate(x):
        if i%64 == 0:
            if i != 0:
                ret.append((in_tmp, out_tmp))
            in_tmp = []
            out_tmp = []
        in_tmp.append(x[0])
        out_tmp.append(x[1])

    return ret


if __name__ == '__main__':
    # create vocabulary files by "train.en" and "train.ja".                                                          
    util.make_vocabs(in_file, voc_file, voc_size)

    # convert word to word_id
    util.apply_vocabs(in_file, wid_file, voc_file)

    datas = []
    with open(wid_file) as f:
        for l in f:
            xs = l.split()
            #tmpdata = xp.zeros(voc_size, dtype=xp.float32)
            tmpdata = [0 for _ in range(voc_size)]
            for x in xs:
                tmpdata[int(x)] += 1
            for x in xs:
                datas.append((tmpdata, x))
    
    # model setup                                                                                                      
    mdl = model.AutoEncoder(voc_size, hidden_size)
    # optimizer set up                                                                                                 
    opt = chainer.optimizers.Adam()
    opt.setup(mdl)
    # use gpu                                                                                                          
    chainer.cuda.get_device(gpu).use()
    mdl.to_gpu()

    for i in range(10):
        all_loss = 0
        batches = create_batch(datas)
        for b in batches:
            mdl.zerograds()
            loss = mdl.forward(b[0], b[1])
            loss.backward()
            opt.update()
            all_loss += loss.data
        print all_loss
