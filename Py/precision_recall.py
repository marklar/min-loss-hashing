#!/usr/bin/env python
"""
Given:
  * set of true neighbors
  * sketch hamming distances (ham-dists.dat)

"""
from util import *
from files import *
from scores import *

from itertools import izip, imap
from collections import namedtuple
import numpy as np
import sys

def f_score(prec, rec, beta=1.0):
    """
    when beta:
        0.5 -> favors precision
        1.0 -> even weighting
        2.0 -> favors recall
    """
    p, r, b2 = prec, rec, beta**2
    res = (1+b2) * (p*r / (b2*p + r))
    if np.isnan(res):
        return 0.0
    else:
        return res

def calc_results(good, total, n_good, n_bits):
    precision = np.zeros(n_bits+1)
    recall    = np.zeros(n_bits+1)
    for i in range(n_bits+1):
        if good[i]:
            g = float(good[i])
            precision[i] = g / total[i]
            recall[i]    = g / n_good
    return precision, recall

def show_counts(good, total, n_good):
    print 'retrieved:'
    print '  good: ', good
    print '  total:', total
    print 'n_good:', n_good

def gather_stats(sim_dist_pairs, cfg):
    # At each radius, store how many were retrieved.
    good  = np.zeros(cfg.n_bits + 1)
    total = np.zeros(cfg.n_bits + 1)
    t = 0
    for s,d in sim_dist_pairs:
        t = tick(t, 100000)
        total[d] += 1
        if s >= cfg.near_thresh: good[d] += 1
    tock()
    # Cumulative.
    total = np.cumsum(total)
    good  = np.cumsum(good)
    n_good = good[cfg.n_bits]
    # show_counts(good, total, n_good)
    return calc_results(good, total, n_good, cfg.n_bits)

def show_stats(precisions, recalls):
    for i in range(len(precisions)):
        print 'radius:', i
        print '  precision:', precisions[i]
        print '  recall:   ', recalls[i]
        print '  f05_score:', f_score(precisions[i], recalls[i], 0.5)
        print '  f1_score: ', f_score(precisions[i], recalls[i])
        print '  f2_score: ', f_score(precisions[i], recalls[i], 2.0)
        print

def estimate_rho(recalls):
    return sum(recalls < 0.3)

#---------------------------------

def calc(cfg):
    cos_sims  = (s['score']    for s in get_sorted_sims(cfg))
    ham_dists = (h['distance'] for h in get_sorted_hams(cfg))

    t = beg('Computing stats')
    sim_dist_pairs = izip(cos_sims, ham_dists)
    precisions, recalls = gather_stats(sim_dist_pairs, cfg)
    end(t)

    show_stats(precisions, recalls)    

    rho = estimate_rho(recalls)
    print 'rho estimate:', rho

if __name__ == '__main__':
    cfg = get_cfg()
    calc(cfg)
