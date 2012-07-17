#!/usr/bin/env python
"""
Read in Forbes FVs.
Assign ID to each feature.
Calculate magnitude for each vector.
Save id-mag-vector to Mongo.
Write feature-id map to json.
"""
from util import *
from files import *

import numpy as np

DESIRED_KEYS = ['lemmapos', 'namedent', 'concepts']
def feature_wt_pairs(doc):
    for feature_type in DESIRED_KEYS:
        for pair in doc['fvs'][feature_type]:
            yield tuple(pair)

def calc_magnitude(xs):
    return np.sqrt(sum(np.array(xs) ** 2))

G_feature_id_map = {}
G_next_feature_id = 0
def get_feature_id(feature):
    """ For each key, assign a unique ID. """
    global G_feature_id_map, G_next_feature_id
    try:
        return G_feature_id_map[feature]
    except KeyError:
        res = G_next_feature_id
        G_feature_id_map[feature] = res
        G_next_feature_id += 1
        return res

def featID_wt_pairs(fv):
    for k,v in fv:
        fid = get_feature_id(k)
        yield fid, v

def forbes_fvs():
    coll = mongo_coll('forbes', 'forbes_110908_featurized_tfidf')
    for doc in coll.find():
        fv = list(featID_wt_pairs(feature_wt_pairs(doc)))
        yield {'_id'       : doc['_id'],
               'magnitude' : calc_magnitude(map(snd, fv)),
               'features'  : sorted(fv) }

def save_fvs():
    coll = mongo_clean_coll('mlh', 'fvs')
    t = 0
    for group in groups_of(forbes_fvs(), 10):
        t = tick(t, 10)
        coll.insert(group)
    tock()
    
#------------------------------------

def make():
    be = beg('Writing FVs')
    save_fvs()
    end(be)
    
    global G_feature_id_map, G_next_feature_id
    print 'num features:', G_next_feature_id
    be = beg('Writing feature-id map')
    json.dump(G_feature_id_map, file(FEATURE_IDS_FN, 'w'))
    end(be)
    del G_feature_id_map

if __name__ == '__main__':
    make()
