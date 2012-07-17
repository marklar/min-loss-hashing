#!/usr/bin/env python
import sys, os

from util import *
import files
import feature_vectors
import cosine_sims
import mean_values
import lsh_W
import sketch
import hamming
import small_sims
import correlation
import precision_recall

#------------------------------------

def get_cfg():
    try:
        docs, bits, near_thresh, quantize = sys.argv[1:5]
        return Cfg(int(docs), int(bits), float(near_thresh), bool(int(quantize)))
    except:
        exe = sys.argv[0].lstrip('./')
        print 'Usage: {0} <NUM DOCS> <NUM BITS> <NEAR THRESH> <QUANTIZE 0/1>'.format(exe)
        sys.exit(1)

def run():
    cfg = get_cfg()

    be = beg('RUNNING')
    if not os.path.exists(files.data_dir):
        os.mkdir(files.data_dir)

    if False:
        # These two never change.  Once done, don't need to repeat.
        t = beg('FEATURE_VECTORS') ;  feature_vectors.make() ;  end(t)
        t = beg('MEAN_VALUES')     ;  mean_values.calc()     ;  end(t)

    # Can take a long time.  No-ops if collection exists (& is non-empty).
    t = beg('COSINE_SIMS') ;  cosine_sims.calc(cfg)   ;  end(t)
    t = beg('LSH_W')       ;  lsh_W.make(cfg.n_bits)  ;  end(t)
    t = beg('SKETCH')      ;  sketch.make(cfg)        ;  end(t)
    if cfg.quantize:
        t = beg('HAMMING')    ;  hamming.calc(cfg)    ;  end(t)
    else:
        t = beg('SMALL_SIMS') ;  small_sims.calc(cfg) ;  end(t)
    t = beg('CORRELATION') ;  correlation.calc(cfg)        ;  end(t)

    # t = beg('PREC_RECALL') ;  precision_recall.calc(cfg)   ;  end(t)
    end(be)

run()

#
# -- validate --
#
# there exist:
#   files:
#      [Data/W.npy, Data/feature-ids.json, Data/mean-vals.npy]
#   DB ('mlh') collections:
#      [u'fvs', u'cos_sims', u'sketches', u'ham_dists']
#      db.cos_sims.count() == db.ham_dists.count()
#      db.cos_sims.count() == sum(range(db.sketches.count()))
#
