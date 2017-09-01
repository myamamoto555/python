# coding:utf-8

import bleu
from collections import defaultdict
import train_util
import corpus_util
import os
import time
import chainer

train_src = "./sample_data/tok/train.en"
train_trg = "./sample_data/tok/train.ja"
dev_src = "./sample_data/tok/dev.en"
dev_trg = "./sample_data/tok/dev.ja"
test_src = "./sample_data/tok/test.en"
test_trg = "./sample_data/tok/test.ja"
src_vocab_path = "./sample_data/vocab/train.en"
trg_vocab_path = "./sample_data/vocab/train.ja"
vocab_dir = "./sample_data/vocab/"
input_dir = "./sample_data/tok/"
wid_dir = "./sample_data/wid/"
result_dir = "./sample_data/result/"
eval_dir = "./sample_data/eval/"
model_dir = "./sample_data/model/"
model_file = None  # load files                                                  
#src_vocab_size = 6637                                                                   
#trg_vocab_size = 8777                                                                    
src_vocab_size = 65536
trg_vocab_size = 65536
embed_size = 512
hidden_size = 512
atten_size = 512
train_batch_size = 64
test_batch_size = 8
max_sample_length = 50
max_generation_length = 80
total_steps = 100000
eval_interval = 100000
save_interval = 1000
gradient_clipping = 2.0
weight_decay = 0.0001
beam_size = 8
gpu = 0


# create batches                                                                                   
train_batches, dev_batches, test_batches = train_util.prepare_data(
    wid_dir+"train.en", wid_dir+"train.ja", wid_dir+"dev.en", wid_dir+"dev.ja",
    wid_dir+"test.en", wid_dir+"test.ja", train_batch_size, test_batch_size,
    max_sample_length, beam_size)


# model setup
mdl = train_util.init_atten_encdec_model(src_vocab_size, trg_vocab_size,
                                         embed_size, hidden_size, atten_size)
opt = train_util.init_optimizer(gradient_clipping, weight_decay, mdl)
train_util.prepare_gpu(gpu, mdl)

print chainer.config.train
with chainer.using_config('train', False):
    print chainer.config.train
    for i in range(35000, 35001, 1000):
        train_util.load_model('./sample_data/model/' + str(i), mdl)
        
        dev_accum_loss, dev_hyps = train_util.test_model(
            mdl, dev_batches, max_generation_length, beam_size)
        train_util.save_hyps(result_dir + 'dev.hyp.%08d' % i, dev_hyps)
        
        test_accum_loss, test_hyps = train_util.test_model(
            mdl, test_batches, max_generation_length, beam_size)
        train_util.save_hyps(result_dir + 'test.hyp.%08d' % i, test_hyps)

# decode & evaluate by blue                                        
files = os.listdir(result_dir)
for fi in files:
    in_fp = result_dir + fi
    out_fp = eval_dir + fi
    voc_fp = vocab_dir + 'train.ja'
    corpus_util.restore_words(in_fp, out_fp, voc_fp)
    # calculate bleu (reference_file, hyp_file, output_file) 
    if 'dev' in fi:
        bleu.calc_bleu_main(input_dir + 'dev.ja', out_fp, out_fp + '.bleu')
    if 'test' in fi:
        bleu.calc_bleu_main(input_dir + 'test.ja', out_fp, out_fp + '.bleu')

