# coding: utf-8

import preprocess
import pickle
import util
import net
import numpy as np

flag = True

src_train_fname = "../data/train.en"
targ_train_fname = "../data/train.ja"
src_dev_fname = "../data/dev.en"
targ_dev_fname = "../data/dev.ja"
src_test_fname = "../data/test.en"
targ_test_fname = "../data/test.ja"


def main():
    if flag == False:
        src_vocs, src_indexes, targ_vocs, targ_indexes = preprocess.train_create(src_train_fname, targ_train_fname)
        with open("src_vocs.pickle", "wb") as f:
            pickle.dump(src_vocs, f)
        with open("src_indexes.pickle", "wb") as f:
            pickle.dump(src_indexes, f)
        with open("targ_vocs.pickle", "wb") as f:
            pickle.dump(targ_vocs, f)
        with open("targ_indexes.pickle", "wb") as f:
            pickle.dump(targ_indexes, f)
    else:
        with open('src_vocs.pickle', 'rb') as f:
            src_vocs = pickle.load(f)
        with open('src_indexes.pickle', 'rb') as f:
            src_indexes = pickle.load(f)
        with open('targ_vocs.pickle', 'rb') as f:
            targ_vocs = pickle.load(f)
        with open('targ_indexes.pickle', 'rb') as f:
            targ_indexes = pickle.load(f)


    src_batch, trg_batch = util.batch_create(np.array(src_indexes), np.array(targ_indexes), 64)

    net.train(src_batch, trg_batch, len(src_vocs), len(targ_vocs), 5)


    #preprocess.dev_create(src_dev_fname, targ_dev_fname)
    #prerpocess.test_create(src_test_fname, targ_test_fname)

if __name__ == '__main__':
    main()
