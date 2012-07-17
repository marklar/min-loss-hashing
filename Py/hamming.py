#!/usr/bin/env python
"""
Compute Hamming SIMILARITIES (i.e. percentage bits same).
Save to file.
"""
from util import *
from scores import *

from bitarray import *
from itertools import combinations

#------------------------------------

def hamming_distance(s1, s2):
    return (s1 ^ s2).count()

def fetch_sketches(cfg):
    coll = sketches_coll(cfg)
    for d in coll.find():
        yield { '_id'    : d['_id'],
                'sketch' : bitarray(mk_ascii(d['sketch'])) }

def distances(sketches):
    return ((hamming_distance(a['sketch'], b['sketch']), a['_id'], b['_id'])
            for a,b in combinations(sketches, 2))

def save_distances(sketches, cfg):
    coll = dists_coll(cfg)
    coll.drop()
    dists = ({'distance':d, 'id1':a, 'id2':b}
             for d,a,b in distances(sketches))

    tm = beg("Inserting (each '.' is 10k records)")
    tk = 0
    for ds in groups_of(dists, 1000):
        tk = tick(tk, 10)
        coll.insert(ds)
    tock()
    end(tm)

    t = beg('Ensuring index')
    coll.ensure_index(SORT_ORDER)
    end(t)

def get_params():
    try:
        docs, bits = sys.argv[1:3]
        return Cfg(int(docs), int(bits), near_thresh=None)
    except:
        exe = sys.argv[0].lstrip('./')
        print 'Usage: {0} <NUM DOCS> <NUM BITS>'.format(exe)
        sys.exit(1)

#------------------------------------

def calc(cfg):
    t = beg('Saving pairwise Hamming distances to DB')
    sketches = fetch_sketches(cfg)
    save_distances(sketches, cfg)
    end(t)

if __name__ == '__main__':
    cfg = get_params()
    calc(cfg)
