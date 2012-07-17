#!/usr/bin/env python
"""
Input:  IDs of two Forbes documents.
Output: Feature overlap between them.

Reads in the weighted feature vectors of both docs,
then finds their overlapping keys.
"""
from files import *

import json, sys

def common_keys(d1, d2):
    for k,v in d1.iteritems():
        if k in d2:
            yield k

def get_vecs(ids):
    for ln in file(WEIGHTED_VECS_FN, 'r'):
        fv = json.loads(ln)
        if fv['id'] in ids:
            yield fv

def common_features(ids):
    """ Assumes: only 2 IDs. """
    a,b = get_vecs(ids)
    return common_keys(a['features'],
                       b['features'])
