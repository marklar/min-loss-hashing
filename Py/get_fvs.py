"""
Exports single function: get_fvs()
"""
from util import *
from scores import *

import random
import numpy as np
import itertools

#--------------------------------------

def get_fvs(cfg):
    t = beg('Getting cases')
    all_sims = get_all_sims(cfg)
    my_assert (lambda : len(all_sims) == sum(range(cfg.n_docs)))
    sims = get_case_sims(all_sims, cfg)
    x1_ids, x2_ids, is_near_v = map(list, zip(*sims))
    x1 = get_vectors(x1_ids)       # nparray of FVs
    x2 = get_vectors(x2_ids)        # ditto
    is_near_v = np.array(is_near_v)  # row vector of bools

    my_assert (lambda : any(map(lambda x: len(x) == cfg.batch_size,
                                [x1, x2, is_near_v])))
    end(t)
    return x1, x2, is_near_v

#--------------------------------------

g_all_sims = None
def get_all_sims(cfg):
    global g_all_sims
    if not g_all_sims:
        t = beg('Fetching from Mongo')
        coll = sims_coll(cfg)
        g_all_sims = list(coll.find({}, {'score':1, 'id1':1, 'id2':1, '_id':0}))
        end(t)
    return g_all_sims

def get_triples(dicts, near_thresh):
    for d in dicts:
        yield d['id1'], d['id2'], (d['score'] >= near_thresh)

def get_case_sims(all_sims, cfg):
    """
    Return cos-sims corresponding observations for this batch.
    """
    near_sims = [s for s in all_sims if s['score'] >= cfg.near_thresh]
    
    def how_many_near():
        "Do not understand this logic!"
        foo = cfg.batch_size * max(cfg.loss_lambda - (len(near_sims) / len(all_sims)), 0)
        return int(min(cfg.batch_size, round(foo)))
    n_near_cases = how_many_near()

    near_sims = random.sample(near_sims, n_near_cases)
    rest_sims = random.sample(all_sims, cfg.batch_size - n_near_cases)
    case_sims = itertools.chain(near_sims, rest_sims)
    return get_triples(case_sims, cfg.near_thresh)

def get_vectors(ids):
    # Don't do this:   return coll.find({'_id': {'$in': ids}})
    # Doesn't work right if there are any repeats in ids.
    # Also, I don't know whether it maintains the proper order!
    coll = mongo_coll('mlh', 'fvs')
    return np.array([coll.find_one({'_id': i}) for i in ids])
