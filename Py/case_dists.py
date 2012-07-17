"""
Training.
"""
from util import *
import numpy as np

#
# Number of bits for which the two FVs have the DIFF sketch bit.
#
def get_dists_vec(SortedDeltas, Loss, cfg):
    """
    Number of bits different between two sketches, using loss-adjusted inference.
    One val per case in minibatch.
    """
    # LOSS-ADJUSTED INFERENCE
    t = beg('dists_vec')
    CumSums = get_CumSums(SortedDeltas, cfg)
    # Loss is small: always between 0.0 and 1.0.
    # Whereas CumSums can be large.
    # Loss may cause almost no change to CumSums.

    # "loss-adjusted": Adjust CumSums by Loss.
    # Idx of max val w/in column of that col's max value.
    # Vec sould be *almost* the same as CumSums.argmax(0),
    # the index at which SortedDeltas switch to negative.
    vec = (CumSums + Loss).argmax(0)
    my_assert (lambda : len(vec) == cfg.batch_size)
    my_assert (lambda : all(0 <= vec) and all(vec <= cfg.n_bits))
    print vec
    end(t)
    return vec

#-------------------------------

def get_CumSums(SortedDeltas, cfg):
    # cumsum(): in cols, each output elem is running total of prevs.
    CumSums = np.cumsum(SortedDeltas, axis=0)
    print 'CumSums';  print CumSums
    # Take cumulative sums rather than just sum of column
    # because we're also going to add in the Loss at each cell.
    my_assert (lambda : CumSums.shape == cfg.shape)
    assert_monotonic_rise_fall(CumSums, cfg)
    
    # We add a row of zeros, to match shape of SortedDeltas.
    RowOfZeros = np.zeros(cfg.batch_size).reshape(1, cfg.batch_size)
    CumSums = np.concatenate((RowOfZeros, CumSums))
    my_assert (lambda : CumSums.shape == (cfg.n_bits+1, cfg.batch_size))
    return CumSums

def assert_monotonic_rise_fall(CumSums, cfg):
    """
    In each col, going from top to bottom...
    the cumsum climbs monotonically toward max,
    then decreases again monotonically.
    """
    global g_assert
    if not g_assert:
        pass
    else:
        max_idxs = CumSums.argmax(0)
        for col_idx in range(cfg.batch_size):
            col = CumSums[:,col_idx]
            for row_idx in range(max_idxs[col_idx]):
                assert col[row_idx] <= col[row_idx + 1]
            for row_idx in range(max_idxs[col_idx], cfg.n_bits-1):
                assert col[row_idx] >= col[row_idx + 1]
