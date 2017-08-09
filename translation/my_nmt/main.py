#coding: utf-8

import blue
from collections import defaultdict
import train_util
import corpus_util
import os

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
src_vocab_size = 6637
trg_vocab_size = 8777
embed_size = 512
hidden_size = 512
atten_size = 512
train_batch_size = 128
test_batch_size = 16
max_sample_length = 50
max_generation_length = 100
total_steps = 100
eval_interval = 100
gradient_clipping = 2.0
weight_decay = 0.0001
gpu = 0


if __name__ == '__main__':
    # create directory
    corpus_util.create_dirs(vocab_dir, wid_dir, result_dir, eval_dir)

    # create vocabulary files by "train.en" and "train.ja".
    corpus_util.make_vocabs(train_src, src_vocab_path, src_vocab_size, 
                            train_trg, trg_vocab_path, trg_vocab_size)

    # convert word to word_id. apply it to "train", "dev" and "test" files.
    for kind in["train", "dev", "test"]:
        for lang in ["en", "ja"]:
            filename = kind + "." + lang
            in_fp = input_dir + filename
            wid_fp = wid_dir + filename
            vocab_path = vocab_dir + "train." + lang
            corpus_util.apply_vocabs(in_fp, wid_fp, vocab_path)
    
    # create batches
    train_batches, dev_batches, test_batches = train_util.prepare_data(
        wid_dir+"train.en", wid_dir+"train.ja", wid_dir+"dev.en", wid_dir+"dev.ja",
        wid_dir+"test.en", wid_dir+"test.ja", train_batch_size, test_batch_size,
        max_sample_length)

    for i in range(1000):
        print len(next(train_batches))
    print aaa

    # model setup
    mdl = train_util.init_atten_encdec_model(src_vocab_size, trg_vocab_size, 
                                             embed_size, hidden_size, atten_size)
    opt = train_util.init_optimizer(gradient_clipping, weight_decay, mdl)
    train_util.prepare_gpu(gpu, mdl)

    #training loop
    trained_samples = 0
    for step in range(1, total_steps + 1):
        trained_samples += train_util.train_step(mdl, opt, train_batches)

        if step % eval_interval == 0:
            step_str = 'Step %d/%d' % (step, total_steps)
            dev_accum_loss, dev_hyps = train_util.test_model(
                mdl, dev_batches, max_generation_length)
            train_util.save_hyps(result_dir + 'dev.hyp.%08d' % step, dev_hyps)

            test_accum_loss, test_hyps = train_util.test_model(
                mdl, test_batches, max_generation_length)
            train_util.save_hyps(result_dir + 'test.hyp.%08d' % step, test_hyps)

    # decode & evaluate by blue
    files = os.listdir(result_dir)
    for fi in files:
        in_fp = result_dir + fi
        out_fp = eval_dir + fi
        voc_fp = vocab_dir + fi
        corpus_util.restor_words(in_fp, out_fp, voc_fp)
