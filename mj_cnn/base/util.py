#coding: utf-8

from collections import defaultdict
import os
import model
import chainer
import chainer.optimizers
import chainer.serializers


def make_vocab(in_fp, out_fp, size):
    freq = defaultdict(int)
    num_lines = 0
    with open(in_fp) as fp:
        for line in fp:
            num_lines += 1
            for word in line.split():
                freq[word] += 1

    freq_sorted = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    num_unk = sum(x[1] for x in freq_sorted[size - 3:])

    with open(out_fp, 'w') as fp:
        fp.write('0\t<unk>\t%d\n' % num_unk)
        fp.write('1\t<s>\t%d\n' % num_lines)
        fp.write('2\t</s>\t%d\n' % num_lines)
        for i, (key, val) in zip(range(3, size), freq_sorted):
            fp.write('%d\t%s\t%d\n' % (i, key, val))


def apply_vocabs(in_fp, out_fp, vocab_path):
    vocab = defaultdict(lambda: '0')
    with open(vocab_path) as vocab_file:
        for line in vocab_file:
            word_id, word, freq = line.split()
            vocab[word] = word_id
    with open(in_fp) as input_file, open(out_fp, 'w') as output_file:
        for line in input_file:
            word_ids = [vocab[w] for w in line.split()]
            output_file.write(' '.join(word_ids))
            output_file.write("\n")


def init_cnn(vocab_size, embed_size):
    mdl = model.cnn(
        vocab_size,
        embed_size)

    return mdl


def init_optimizer(gradient_clipping, weight_decay, mdl):
    opt = chainer.optimizers.Adam()
    opt.setup(mdl)
    opt.add_hook(chainer.optimizer.GradientClipping(gradient_clipping))
    opt.add_hook(chainer.optimizer.WeightDecay(weight_decay))

    return opt


def prepare_gpu(gpu, mdl):
    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()
        mdl.to_gpu()
        model.use_gpu = True
