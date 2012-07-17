#!/usr/bin/env python
"""
"""
from files import *
from util import *
from scores import *

from itertools import count, izip
import numpy as np
import matplotlib.pyplot as plt
import sys
import pymongo

#------------------------------------------

def pearson_correlation(xs, ys):
    """
    Input: 2 sequences of numbers.
    Return: -1.0 <= coeff <= 1.0
    """
    if type(xs) != np.ndarray:
        xs, ys = np.array(xs), np.array(ys)
    assert len(xs) == len(ys)
    num = len(xs)
    x_sum, y_sum = sum(xs), sum(ys)
    denom = np.sqrt(
        # xs ** 2: element-wise operations.
        (sum(xs ** 2) - (x_sum ** 2) / num) *
        (sum(ys ** 2) - (y_sum ** 2) / num) )
    if denom == 0:
        return 0
    else:
        # xs * ys: pairwise operations.
        numer = sum(xs * ys) - (x_sum * y_sum / num)
        return numer / denom

def validate_input(xs, ys):
    "Tuples"
    xs, ys = list(xs), list(ys)
    # Same number...
    assert len(xs) == len(ys)
    # ...and order!
    assert all(a == x and b == y
               for (_,a,b), (_,x,y)
               in izip(xs, ys))

def scores_pearson(xs, ys):
    """
    Input: 2 lists (generators) of triples: (score, id1, id2).
    Return: -1.0 <= coeff <= 1.0
    """
    validate_input(xs, ys)
    # We care about only the first col (score).
    def fsts(xs): return [x[0] for x in xs]
    # plot_scores(sims, hams)
    return pearson_correlation(fsts(xs), fsts(ys))

def get_ranks(scores):
    """
    To each pair of IDs, assign a rank (rather than score).
    Then find the correlation between the ranks.
    Ranks (and line nums) start at 0.
    """
    score_w_line_num = zip(map(fst, scores), count())
    ranked_line_nums = map(snd,
                           sorted(score_w_line_num, reverse=True))
    line_num_w_rank = sorted(zip(ranked_line_nums, count()))
    return map(snd, line_num_w_rank)

# deprecated
def ranks_pearson(sims, hams):
    t = beg('Gathering sim ranks')
    sim_ranks = get_ranks(sims)
    end(t)
    t = beg('Gathering ham ranks')
    ham_ranks = get_ranks(hams)
    end(t)
    
    t = beg('Calculating _rank_ correlation')
    def f_ary(ns): return np.array(ns) / 100.0
    sim_ranks = f_ary(sim_ranks)
    ham_ranks = f_ary(ham_ranks)
    pcc = pearson_correlation(sim_ranks, ham_ranks)
    end(t)
    return pcc
    

#------------------------------------------------

def plot_scores(sim_scores, ham_scores):
    num = 1000
    plt.plot(fsts(sim_scores)[:num],
             fsts(ham_scores)[:num],
             'bo')  # blue dots
    plt.xlabel('sim score')
    plt.ylabel('ham score')
    plt.show()

def plot_sims_hams(sims, hams):
    scores = map(fst, sims)
    plt.plot(scores, 'bo')
    plt.ylabel('score')
    plt.xlabel('line number')
    plt.show()
    sys.exit()

def plot_ranks(sim_ranks, ham_ranks):
    num = 1000
    plt.plot(sim_ranks[:num], ham_ranks[:num], 'bo')
    plt.xlabel('sim rank')
    plt.ylabel('ham rank')
    plt.show()

#------------------------------------------------

def sim_to_tuple(s):
    return s['score'], s['id1'], s['id2']
def ham_to_tuple(h):
    return h['distance'], h['id1'], h['id2']

def read_scores(cfg):
    t = beg('Reading in data')
    sims = get_sorted_sims(cfg)
    sims = [sim_to_tuple(s) for s in sims]
    if cfg.quantize:
        hams = get_sorted_hams(cfg)
        hams = [ham_to_tuple(h) for h in hams]
        # plot_sims_hams(sims, hams)
        end(t)
        return sims, hams
    else:
        smalls = get_sorted_small_sims(cfg)
        smalls = [sim_to_tuple(h) for h in smalls]
        end(t)
        return sims, smalls

#------------------------------------------------

def get_cfg():
    try:
        docs, bits, quantize = sys.argv[1:]
        return Cfg(int(docs), int(bits), None, bool(int(quantize)))
    except:
        exe = sys.argv[0].lstrip('./')
        print 'Usage: {0} <NUM DOCS> <NUM BITS> <QUANTIZE>'.format(exe)
        sys.exit(1)

def calc(cfg):
    xs, ys = read_scores(cfg)
    t = beg('Calculating _score_ correlation')
    score_pcc = scores_pearson(xs, ys)
    end(t)
    print 'Score PCC:', score_pcc

    #-- ranks --
    # rank_pcc = ranks_pearson(sims, hams)
    # print 'Rank PCC:', rank_pcc
    # plot_ranks(sim_ranks, ham_ranks)

if __name__ == '__main__':
    cfg = get_cfg()
    calc(cfg)
