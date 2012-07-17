"""
Exports: calc_dot_prods(), get_Wx1_Wx2(), get_W()
"""
from sketch import mk_centered_x
from files import *
from util import *

import numpy as np

def get_W(cfg):
    t = beg('Loading W from file')
    fn = hyperplanes_fn(cfg.n_bits)
    W = np.load(fn)
    my_assert (lambda : W.shape[0] == cfg.n_bits)
    end(t)
    return W

#--------------------------------

def get_Wx1_Wx2(W, x1, x2, cfg):
    """
    Each cell is the dot product (i.e. sum of products) between:
       + a hyperplane     (row from W)
       + a feature vector (col from X)
    Each col is the raw score, before taking sign() [to make sketch].
    """
    t = beg('Loading mean vals')
    mean_vals = np.load(MEAN_VALS_FN)
    my_assert (lambda : len(mean_vals) == W.shape[1])
    end(t)
    t = beg('Calculating Wx matrices')
    Wx1 = calc_Wx(W, x1, mean_vals, cfg)
    Wx2 = calc_Wx(W, x2, mean_vals, cfg)
    end(t)
    # print 'Wx1';  print Wx1;  print 'Wx2';  print Wx2
    return Wx1, Wx2

def calc_Wx(W, vectors, mean_vals, cfg):
    X = make_X(vectors, mean_vals)
    WX = np.dot(W, X)
    my_assert (lambda : WX.shape == cfg.shape)
    return WX

def make_X(vectors, mean_vals):
    arrays = [mk_centered_x(v['features'], mean_vals) for v in vectors]
    return np.array(arrays).transpose()

#---------------------------

def calc_dot_prods(Wx1, Wx2, cfg):
    # We have 2 mtxs.  Both have dot products: FVs w/ HPs.
    # Each case is a pair of FVs.
    # Wx1 is for the 1st FV per pair.
    # Wx2 is for the 2nd.
    #
    # Big abs(dot-prod) -> FV far from HP.
    # If the dot-prods (DPs) for both FVs in a pair have the same sign,
    # that means they're on the same side of the HP.
    #
    # If abs(DP1 + DP2) > abs(DP1 - DP2) -> on same side of HP.
    #
    # If they're on the same side, and the abs(sum of their dot-prods)
    # is large, then...
    
    # N.B.: Matlab's indices are from 1.  Numpy's are from 0.

    # Idxs... Average: positive (0) or negative (1)?
    Sums,  SumIdxs  = calc_dps(Wx1 + Wx2, -(Wx1 + Wx2), cfg)
    # Idxs... Which is larger: x1 (0) or x2 (1)?    (Strictly, not in absolute.)
    Diffs, DiffIdxs = calc_dps(Wx1 - Wx2, Wx2 - Wx1, cfg)
    t = beg('SortedDeltas')
    Deltas = Diffs - Sums
    del Diffs
    del Sums
    my_assert (lambda : Deltas.shape == cfg.shape)
    # delta: between diff and sum
    #
    # Sign of delta -- Is HP between FVs or to same side?
    #    - positive <- FVS on opposite sides of HP (i.e. HP between them)
    #    - negative <- FVs on same side of HP
    #
    # Abs(delta) -- proportional to abs(distance) of HP to nearer FV:
    #    - small <- HP close to nearer FV
    #    - large <- HP far from nearer FV
    #
    # Large positive: further apart, HP closer to mean.
    #
    # Sorted from [1] between & far away ->  [2] same side & far away.
    #             [1] Most discrimitating -> [2] least discriminating.
    #
    print 'Deltas';  print Deltas
    SortedDeltas = rev_sort_cols(Deltas)
    my_assert (lambda : SortedDeltas.shape == cfg.shape)
    #
    # Tells you *which* HPs were more/less discriminating for each case.
    SortedDeltaIdxs = np.flipud(Deltas.argsort(axis=0))
    print 'SortedDeltaIdxs';  print SortedDeltaIdxs
    # del Deltas
    my_assert (lambda : SortedDeltaIdxs.shape == cfg.shape)
    end(t)
    
    print 'SortedDeltas:';  print SortedDeltas
    return SortedDeltas, SortedDeltaIdxs, SumIdxs, DiffIdxs

#---------------------------

def calc_dps(First, Second, cfg):
    t = beg('calc_Dps')
    tensor = np.array([First, Second])
    my_assert (lambda : tensor.shape == (2, cfg.n_bits, cfg.batch_size))
    Vals = tensor.max(0)
    my_assert (lambda : Vals.shape == cfg.shape)
    Idxs = tensor.argmax(0)
    my_assert (lambda : Idxs.shape == cfg.shape)
    idxs = Idxs.flat        
    my_assert (lambda : all((idxs == 0) + (idxs == 1)))
    end(t)
    return Vals, Idxs

def rev_sort_cols(Mtx):
    """
    np.flipud() -- inverts matrix
    No way to specify sort direction.  Instead, sort then flip.
    """
    return np.flipud(np.sort(Mtx, axis=0))

