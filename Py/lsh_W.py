#!/usr/bin/env python
"""
Setup:
  Determine p: number of input dimensions.
  Select q: number of bits in sketch (e.g. 128).

Create matrix W (size: q*(p+1)).
  Random floats centered at 0.
  Final column: all 0s.
Save to file.
"""
from util import *
from files import *

import json, sys
import numpy as np

#--------------------

def get_p():
    """ From feature-ids mapping. """
    return len( np.load(MEAN_VALS_FN) )

def get_q():
    try:
        return int(sys.argv[1])
    except:
        print "Usage: lsh_mk_W.py <NUM BITS>"
        sys.exit(1)
        
#--------------------

def mk_W_normalized(q,p):
    return normalize_matrix( mk_W_unnormalized(q,p) )

#--------------------

def mk_W_unnormalized(q,p):
    """ 0-centered float32s. """
    return np.random.randn(q * p).reshape(q, p).astype(np.float32)

#--------------------

def make(q):
    t = beg('Getting p')
    p = get_p()
    end(t)
    print 'p:', p
    print 'q:', q
    
    t = beg('Making W')
    # In the paper, they normalize the hyperplanes.
    # (They also re-normalize after every iteration of gradient descent.)
    # In terms of Pearson correlation, it makes zero difference
    # whether one normlizes or not.  There must be a purpose: what is it?

    normalize_p = False
    if normalize_p:
        W = mk_W_normalized(q,p)
    else:
        W = mk_W_unnormalized(q,p)
        
    assert W.shape == (q,p)
    end(t)

    # print W

    t = beg('Saving W to file')
    fn = hyperplanes_fn(q)
    np.save(fn, W)
    end(t)

if __name__ == '__main__':
    q = get_q()
    make(q)
