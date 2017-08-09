#coding: utf-8

import util


train_src = "./sample_data/wid/train.en"
train_trg = "./sample_data/wid/train.ja"
dev_src = "./sample_data/wid/dev.en"
dev_trg = "./sample_data/wid/dev.ja"
test_src = "./sample_data/wid/test.en"
test_trg = "./sample_data/wid/test.ja"
train_batch_size = 128
test_batch_size = 16
max_sample_length = 100


if __name__ == '__main__':
    train_batches, dev_batches, test_batches = util.prepare_data(
        train_src, train_trg, dev_src, dev_trg,
        test_src, test_trg, train_batch_size, test_batch_size,
        max_sample_length)
