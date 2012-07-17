from util import *
import pymongo

ASC = pymongo.ASCENDING
SORT_ORDER = [('id1', ASC), ('id2', ASC)]

#---------------
# all records
# This is specifically for when you want them sorted,
# which is important for determining correlation,
# but not for anything else.
# 
def get_sorted_sims(cfg):
    coll = sims_coll(cfg)
    return coll.find().sort(SORT_ORDER)

def get_sorted_small_sims(cfg):
    coll = small_sims_coll(cfg)
    return coll.find().sort(SORT_ORDER)

def get_sorted_hams(cfg):
    coll = dists_coll(cfg)
    return coll.find().sort(SORT_ORDER)

#----------------

def small_sims_coll(cfg):
    name = 'small_similarities_d{0}'.format(cfg.n_docs)
    return mongo_coll('mlh', name)

def sims_coll(cfg):
    name = 'cosine_similarities_d{0}'.format(cfg.n_docs)
    return mongo_coll('mlh', name)

def dists_coll(cfg):
    name = 'hamming_distances_d{0}_q{1}'.format(cfg.n_docs, cfg.n_bits)
    return mongo_coll('mlh', name)

def sketches_coll(cfg):
    name = 'sketches_d{0}_q{1}'.format(cfg.n_docs, cfg.n_bits)
    return mongo_coll('mlh', name)

def small_embeddings_coll(cfg):
    name = 'small_embeddings_d{0}_q{1}'.format(cfg.n_docs, cfg.n_bits)
    return mongo_coll('mlh', name)
