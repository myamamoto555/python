#coding: utf-8

import random
from collections import defaultdict


def _read_samples(sent_filepath, ans_filepath, bos_id, eos_id):
    ret = []
    with open(sent_filepath) as sfp, open(ans_filepath) as afp:
        for sent, ans in zip(sfp, afp):
            sent = sent.split("\n")[0]
            ans = ans.split("\n")[0]
            s = [bos_id] + [int(x) for x in sent.split()] + [eos_id]
            a = float(ans)
            ret.append((s,a))
    return ret


def _split_samples(samples, batch_size):
    batches = []
    for i in range(0, len(samples), batch_size):
        batches.append(samples[i : i + batch_size])
    return batches


def _make_batch(samples, pad_id):
    batch_size = len(samples)
    max_length = max(len(sample[0]) for sample in samples)
    sent_batch = [[pad_id] * max_length for _ in range(batch_size)]
    ans_batch = []
    for i, (sent, ans) in enumerate(samples):
        for j, w in enumerate(sent):
            sent_batch[i][j] = w
        ans_batch.append(ans)
    return sent_batch, ans_batch


def generate_samples(
        sent_filepath,
        ans_filepath,
        bos_id,
        eos_id):
    samples = _read_samples(sent_filepath, ans_filepath, bos_id, eos_id)
    return samples


def generate_batches(samples, batch_size, pad_id):
    random.shuffle(samples)
    spsamples = _split_samples(samples, batch_size)
    random.shuffle(spsamples)
    batches = []
    for sp in spsamples:
        sent, ans = _make_batch(sp, pad_id)
        batches.append((sent, ans))
    return batches


def generate_test_batches(
        sent_filepath,
        ans_filepath,
        batch_size,
        bos_id,
        eos_id,
        pad_id):
    samples = _read_samples(sent_filepath, ans_filepath, bos_id, eos_id)
    spsamples = _split_samples(samples, batch_size)
    batches = []
    for sp in spsamples:
        sent, ans = _make_batch(sp, pad_id)
        batches.append((sent, ans))
    return batches

