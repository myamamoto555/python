# coding: utf-8

import math
from argparse import ArgumentParser
from collections import defaultdict


def get_bleu_stats(ref, hyp, N=4):
    stats = defaultdict(int, {'rl': len(ref), 'hl': len(hyp)})
    N = len(hyp) if len(hyp) < N else N
    for n in range(N):
        matched = 0
        possible = defaultdict(int)
        for k in range(len(ref) - n):
            possible[tuple(ref[k : k + n + 1])] += 1
        for k in range(len(hyp) - n):
            ngram = tuple(hyp[k : k + n + 1])
            if possible[ngram] > 0:
                possible[ngram] -= 1
                matched += 1
        stats['d' + str(n + 1)] = len(hyp) - n
        stats['n' + str(n + 1)] = matched
    return stats


def calculate_bleu(stats, N=4):
    np = 0.0
    for n in range(N):
        nn = stats['n' + str(n + 1)]
        if nn == 0:
            return 0.0
        dd = stats['d' + str(n + 1)]
        np += math.log(nn) - math.log(dd)
    bp = 1.0 - stats['rl'] / stats['hl']
    if bp > 0.0: bp = 0.0
    return math.exp(np / N + bp)


def calc_bleu_main(ref, hyp, out):
    stats = defaultdict(int)
    for ref, hyp in zip(open(ref), open(hyp)):
        for k, v in get_bleu_stats(ref.split(), hyp.split()).items():
            stats[k] += v
    fout = open(out, 'w')
    fout.write(str(calculate_bleu(stats)))
    fout.close()


