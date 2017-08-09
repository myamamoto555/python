#coding: utf-8

import batch

PAD_ID = -1
BOS_ID = 1
EOS_ID = 2


def prepare_data(train_src, train_trg, dev_src, dev_trg, 
                 test_src, test_trg, train_batch_size, test_batch_size, 
                 max_sample_length):
    train_batches = batch.generate_train_batch(
        train_src, train_trg,
        PAD_ID, BOS_ID, EOS_ID,
        train_batch_size,
        max_sample_length)


    dev_batches = list(batch.generate_test_batch(
        dev_src, dev_trg,
        PAD_ID, BOS_ID, EOS_ID,
        test_batch_size))

    test_batches = list(batch.generate_test_batch(
        test_src, test_trg,
        PAD_ID, BOS_ID, EOS_ID,
        test_batch_size))

    return train_batches, dev_batches, test_batches
