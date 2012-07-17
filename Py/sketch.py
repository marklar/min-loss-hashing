#!/usr/bin/env python
"""
Index:
  Read in input vectors singly.
  Hash them against W.
  Save to file.
"""
from files import *
from util import *
from scores import *

import sys
import json
import numpy as np
from bitarray import *

DT = np.float32  # uses less space than float64 (default)

def mk_sketch(x, W, cfg):
    """ Given centered input ndarray, create sketch. """
    if cfg.quantize:
        bools = [np.dot(x, row) >= 0.0 for row in W]
        return bitarray(bools)
    else:
        x = x.reshape(-1,1)
        res = np.dot(W, x).flatten()
        return normalize_array(res)
        
def mk_centered_x(vec, mean_vals):
    """ Turn a vector into a centered input ndarray. """
    return mk_array(vec, len(mean_vals)).astype(DT) - mean_vals

def mk_sketches(fvs, W, mean_vals, cfg):
    """ Index (create sketches for) all the feature vectors. """
    for vec in fvs:
        x = mk_centered_x(vec['features'], mean_vals)
        yield vec['_id'], mk_sketch(x, W, cfg)

def save_sketches(sketches, cfg):
    if cfg.quantize:
        coll = sketches_coll(cfg)
        sk_dicts = (dict(_id=i, sketch=s.to01()) for i,s in sketches)
    else:
        coll = small_embeddings_coll(cfg)
        sk_dicts = (dict(_id=i, sketch=list(s)) for i,s in sketches)
    tk = 0
    for group in groups_of(sk_dicts, 100):
        tk = tick(tk, 1)
        coll.insert(group)
    tock()

#--------------------------

def get_cfg():
    try:
        docs, bits, quantize = sys.argv[1:]
        print docs
        print bits
        print quantize
        cfg = Cfg(int(docs), int(bits), None, bool(int(quantize)))
        print cfg
        return cfg
    except:
        exe = sys.argv[0].lstrip('./')
        print 'Usage: {0} <NUM DOCS> <NUM BITS> <QUANTIZE>'.format(exe)
        sys.exit(1)

def make(cfg):
    t = beg('Loading W')
    fn = hyperplanes_fn(cfg.n_bits)
    W = np.load(fn)
    end(t)
    print 'n_bits:', W.shape[1]
    
    t = beg('Loading mean vals')
    mean_vals = np.load(MEAN_VALS_FN)
    end(t)

    t = beg('Selecting {0} largest fvs'.format(cfg.n_docs))
    big_fvs = n_biggest_fvs(cfg.n_docs)
    end(t)
    
    t = beg('Saving sketches 100 at a time')
    sketches = mk_sketches(big_fvs, W, mean_vals, cfg)
    save_sketches(sketches, cfg)
    end(t)

if __name__ == '__main__':
    cfg = get_cfg()
    make(cfg)
