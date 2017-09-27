#coding:utf-8

import util
import batch
import model
import time
import chainer

train_file = './data/train.sent'
train_ans_file = './data/train.ans'
dev_file = './data/dev.sent'
dev_ans_file = './data/dev.ans'
test_file = './data/test.sent'
test_ans_file = './data/test.ans'
train_wid = './data/train.wid'
dev_wid = './data/dev.wid'
test_wid = './data/test.wid'

voc_file = './data/vocab.txt'
voc_size = 33624
batch_size = 64
epoch = 100
embed_size = 200

gradient_clipping = 5
weight_decay = 0.0001
PAD_ID = -1
BOS_ID = 1
EOS_ID = 2
gpu = 0

def evaluate(tb):
    # evaluate                                                                                                                
    acc = 0
    for tb in tb:
        x_list, t_list = tb
        acc += mdl.test(x_list, t_list)
        
    return acc


if __name__ == '__main__':
    # create vocab file
    util.make_vocab(train_file, voc_file, voc_size)
    # convert word to id
    util.apply_vocabs(train_file, train_wid, voc_file)
    util.apply_vocabs(dev_file, dev_wid, voc_file)
    util.apply_vocabs(test_file, test_wid, voc_file)

    # create samples
    train_samples = batch.generate_samples(train_wid, train_ans_file, 
                                           BOS_ID, EOS_ID)
    dev_batches = batch.generate_test_batches(dev_wid, dev_ans_file, batch_size,
                                              BOS_ID, EOS_ID, PAD_ID)
    test_batches = batch.generate_test_batches(test_wid, test_ans_file, batch_size,
                                               BOS_ID, EOS_ID, PAD_ID)

    # model setup
    mdl = util.init_cnn(voc_size, embed_size)
    opt = util.init_optimizer(gradient_clipping, weight_decay, mdl)
    util.prepare_gpu(gpu, mdl)

    # training loop
    for ep in range(epoch):
        start = time.time()
        batches = batch.generate_batches(train_samples, batch_size, PAD_ID)
        for b in batches:
            with chainer.using_config('train', True):
                x_list, t_list = b
                mdl.zerograds()
                loss = mdl.train(x_list, t_list)
                loss.backward()
                opt.update()
            with chainer.using_config('train', False):
                #print chainer.config.train
                dev_acc = evaluate(dev_batches)/ float(2000)
                test_acc = evaluate(test_batches) / float(626)
                print dev_acc, test_acc
        end = time.time()
        #print end - start

