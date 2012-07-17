#!/usr/bin/env python
"""
"""
from util import *
from scores import *

from itertools import combinations

#-------------------------

def dot_product(a,b):
    """ Sigma(a[i] * b[i]). """
    weight_pairs = ord_pairs(a['features'], b['features'])
    return sum(w1 * w2 for w1,w2 in weight_pairs)

def cosine_similarity(a,b):
    return dot_product(a,b) / (a['magnitude'] * b['magnitude'])

def calc_sims(wfvs):
    return ((cosine_similarity(a,b), a['_id'], b['_id'])
            for a,b in combinations(wfvs, 2))

#-------------------------

def save_sims(wfvs, n_big):
    coll = sims_coll(Cfg(n_big, None, None, None))
    if False and coll.count() != 0:
        # If collection already exists, just no-op.
        print 'Skipping.'
        return
    else:
        coll.drop()
        sims = ({'score':s, 'id1':a, 'id2':b} for s,a,b in calc_sims(wfvs))
        be = beg("Inserting (each '.' is 10k records)")
        tk = 0
        for ss in groups_of(sims, 1000):
            tk = tick(tk, 10)
            coll.insert(ss)
        tock()
        end(be)
        coll.ensure_index(SORT_ORDER)

#-------------------------

def calc(cfg):
    t = beg('Selecting {0} largest fvs'.format(cfg.n_docs))
    big_fvs = n_biggest_fvs(cfg.n_docs)
    end(t)

    n_sims = sum(range(cfg.n_docs))
    t = beg('Calculating {0} sims for BIG docs'.format(n_sims))
    save_sims(big_fvs, cfg.n_docs)
    del big_fvs
    end(t)

if __name__ == '__main__':
    cfg = get_docs_cfg()
    calc(cfg)
