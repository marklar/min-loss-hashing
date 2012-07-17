#!/usr/bin/env python
"""
Calculate pairwise cosine similarities
for small embeddings.
"""
from util import *
from scores import *

from itertools import combinations

#-------------------------

def calc_sims(embeddings):
    return ((np.dot(a['sketch'], b['sketch']), a['_id'], b['_id'])
            for a,b in combinations(embeddings, 2))

#-------------------------

def save_smalls(embeddings, cfg):
    coll = small_sims_coll(cfg)
    if False and coll.count() != 0:
        # If collection already exists, just no-op.
        print 'Skipping.'
        return
    else:
        coll.drop()
        sims = ({'score':s, 'id1':a, 'id2':b} for s,a,b in calc_sims(embeddings))
        be = beg("Inserting (each '.' is 10k records)")
        tk = 0
        for ss in groups_of(sims, 1000):
            tk = tick(tk, 10)
            coll.insert(ss)
        tock()
        end(be)
        coll.ensure_index(SORT_ORDER)

#-------------------------

def get_cfg():
    try:
        docs, bits = sys.argv[1:3]
        return Cfg(int(docs), int(bits), None)
    except:
        exe = sys.argv[0].lstrip('./')
        print 'Usage: {0} <NUM DOCS> <NUM BITS>'.format(exe)
        sys.exit(1)

def calc(cfg):
    # Just grab them all.
    t = beg('Fetching {0} small embeddings'.format(cfg.n_docs))
    coll = small_embeddings_coll(cfg)
    embeddings = coll.find().limit(cfg.n_docs)
    end(t)

    # rename 'sketch' to something more appropriate ("weights"?)
    embeddings = [ dict(_id=doc['_id'], sketch=np.array(doc['sketch']))
                   for doc in embeddings ]
    print len(embeddings)

    n_sims = sum(range(cfg.n_docs))
    t = beg('Calculating {0} sims for BIG docs'.format(n_sims))
    save_smalls(embeddings, cfg)
    end(t)

if __name__ == '__main__':
    cfg = get_cfg()
    calc(cfg)
