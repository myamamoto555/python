#coding: utf-8

import random
import model
from collections import defaultdict


def _read_single_samples(filepath, bos_id, eos_id):
    with open(filepath) as fp:
        for line in fp:
            yield [bos_id] + [int(x) for x in line.split()] + [eos_id]


def _read_parallel_samples(src_filepath, trg_filepath, bos_id, eos_id):
    return zip(
        _read_single_samples(src_filepath, bos_id, eos_id),
        _read_single_samples(trg_filepath, bos_id, eos_id))


def _filter_samples(samples, max_sample_length):
    def _pred_max(x):
        return len(x[0]) <= max_sample_length and len(x[1]) <= max_sample_length
    samples = filter(_pred_max, samples)
    return samples


def _arrange_samples(samples):
    buckets = defaultdict(lambda: [])
    for src, trg in samples:
        buckets[len(src)].append((src, trg))
    for key in sorted(buckets):
        samples_in_bucket = buckets[key]
        random.shuffle(samples_in_bucket)
        for sample in samples_in_bucket:
            yield sample


def _split_samples(samples, batch_size):
    batches = []
    for i in range(0, len(samples), batch_size):
        batches.append(samples[i : i + batch_size])
    return batches


def _make_batch(samples, pad_id):
    batch_size = len(samples)
    max_src_length = max(len(sample[0]) for sample in samples)
    max_trg_length = max(len(sample[1]) for sample in samples)
    src_batch = [[pad_id] * batch_size for _ in range(max_src_length)]
    trg_batch = [[pad_id] * batch_size for _ in range(max_trg_length)]
    for i, (src_sample, trg_sample) in enumerate(samples):
        for j, w in enumerate(src_sample):
            src_batch[j][i] = w
        for j, w in enumerate(trg_sample):
            trg_batch[j][i] = w
    return src_batch, trg_batch


def generate_train_batch(
        src_filepath,
        trg_filepath,
        pad_id,
        bos_id,
        eos_id,
        batch_size,
        max_sample_length):

    while True:
        samples = _read_parallel_samples(src_filepath, trg_filepath, bos_id, eos_id)
        samples = _filter_samples(samples, max_sample_length)
        samples = _arrange_samples(samples)
        samples = list(samples)
        batches = _split_samples(samples, batch_size)
        random.shuffle(batches)
        for batch in batches:
            yield _make_batch(batch, pad_id)


def generate_test_batch(
        src_filepath,
        trg_filepath,
        pad_id,
        bos_id,
        eos_id,
        batch_size):
    samples = list(
        _read_parallel_samples(src_filepath, trg_filepath, bos_id, eos_id))
    batches = _split_samples(samples, batch_size)
    for batch in batches:
        yield _make_batch(batch, pad_id)


def batch_to_samples(batch, eos_id):
    samples = [list(x) for x in zip(*batch)]
    samples = [x[ : x.index(eos_id)] for x in samples]
    return samples
