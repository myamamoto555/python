#coding: utf-8
import chainer
import chainer.optimizers
import chainer.serializers
import batch
import model


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


def init_atten_encdec_model(src_vocab_size, trg_vocab_size,
                            embed_size, hidden_size, atten_size):
    mdl = model.AttentionEncoderDecoder(
        src_vocab_size,
        trg_vocab_size,
        embed_size,
        hidden_size,
        atten_size)

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


def train_step(mdl, opt, train_batch_gen):
    x_list, t_list = next(train_batch_gen)
    mdl.zerograds()
    loss = mdl.forward_train(x_list, t_list)
    loss.backward()
    opt.update()

    return len(x_list[0])


def test_model(mdl, batches, limit):
    accum_loss = 0.0
    hyps = []
    for x_list, t_list in batches:
        accum_loss += len(x_list[0]) * float(mdl.forward_train(x_list, t_list).data)
        z_list = mdl.forward_test(x_list, BOS_ID, EOS_ID, limit)
        hyps.extend(batch.batch_to_samples(z_list, EOS_ID))
    return accum_loss, hyps


def save_hyps(filename, hyps):
    with open(filename, 'w') as fp:
        for hyp in hyps:
            fp.write(' '.join(str(x) for x in hyp)+"\n")

