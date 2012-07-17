#!/usr/bin/env python
"""
Find mean val for each feature.
Useful for centering our hyperplanes for LSH.
"""
from files import *
from util import *

import json
import numpy as np

def get_n_dimensions():
    f2id = json.load(file(FEATURE_IDS_FN))
    return len(f2id)

def mean_vals(n_dimensions):
    sums = np.zeros(n_dimensions)
    n_fvs = 0
    tk = 0
    coll = mongo_coll('mlh', 'fvs')
    for fv in coll.find():
        n_fvs += 1
        for feat,wt in fv['features']: sums[feat] += wt
        tk = tick(tk, 1000)
    tock()
    return sums / n_fvs

#-----------------------------------

def calc():
    be = beg('Getting number of dimensions')
    n_dim = get_n_dimensions()
    end(be)
    
    be = beg('Calculating mean vals')
    mvs = mean_vals(n_dim)
    end(be)
    np.save(MEAN_VALS_FN, mvs)

if __name__ == '__main__':
    calc()
