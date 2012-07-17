#!/usr/bin/env python
"""
"""
from pymongo import Connection

DESIRED_KEYS = ['lemmapos', 'namedent', 'concepts']

def get_db_collection(db_name, coll_name):
    con = Connection()  # ('localhost', 27017)
    db = con[db_name]
    return db[coll_name]

def feature_wt_pairs(forbes_doc):
    for feature_type in DESIRED_KEYS:
        for pair in doc['fvs'][feature_type]:
            yield tuple(pair)

fvs = get_db_collection('forbes', 'forbes_110908_featurized_tfidf')
doc = fvs.find_one()
for f,w in feature_wt_pairs(doc):
    print f, w
