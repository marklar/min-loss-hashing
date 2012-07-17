#!/usr/bin/env python
"""
Given:
  - batch_size: total number of pairs to get
  - near_lambda: threshold for nearness
  - loss_lambda: min fraction of pairs that should be near

Return 3 matrices:
  - X1: one of two instances from a pair
  - X2: the other instance from a pair
  - IsNear: {0,1}, whether pair is near (1) or not (0).
"""
from util import *
from scores import *
# --- specific to training ---
from get_fvs import get_fvs
from calc_dot_prods import calc_dot_prods, get_Wx1_Wx2, get_W
from case_dists import get_dists_vec

import sys
import numpy as np
from matplotlib.mlab import find

from collections import namedtuple
TrainCfg = namedtuple('TrainCfg', ['n_docs', 'n_bits', 'batch_size', 'near_thresh', 'rho', 'loss_lambda', 'shape'])

def get_params():
    n_bits = 8
    batch_size = 100
    return TrainCfg(100, n_bits, batch_size, 0.2, 2, 0.5, (n_bits, batch_size))
    # NOT IN USE...
    try:
        n_docs, n_bits, batch_size, near_thresh, rho, loss_lambda = sys.argv[1:]
        return TrainCfg(int(n_docs), int(n_bits), int(batch_size),
                        float(near_thresh), int(rho), float(loss_lambda),
                        (int(n_bits), int(batch_size)))
    except:
        exe = sys.argv[0].lstrip('./')
        print 'usage: {0} N_DOCS N_BITS BATCH_SIZE(10000) NEAR_THRESH RHO LOSS_LAMBDA(0.5)'.format(exe)
        sys.exit(0)

#------------------------------------

def mk_Loss(cfg, is_near_v):
    t = beg('Loss')
    n_pos_ham = cfg.n_bits + 1   # num possible Hamming distances

    x = n_pos_ham - cfg.rho
    neigh_loss = np.concatenate([np.zeros(cfg.rho), np.arange(x)]) / x
    y = cfg.rho + 1
    non_neigh_loss = np.concatenate([np.arange(y,0,-1), np.zeros(cfg.n_bits-cfg.rho)]) / y
    my_assert (lambda : len(neigh_loss) == n_pos_ham)
    my_assert (lambda : len(non_neigh_loss) == n_pos_ham)

    # For each pair of corresponding cells, one should be 0.
    my_assert (lambda : not np.any(neigh_loss * non_neigh_loss))

    NeighLoss = np.kron(is_near_v,
                        neigh_loss.reshape(n_pos_ham, 1))
    my_assert (lambda : NeighLoss.shape == (n_pos_ham, cfg.batch_size) )

    NonNeighLoss = np.kron(is_near_v == False,
                           non_neigh_loss.reshape(n_pos_ham, 1))
    my_assert (lambda : NonNeighLoss.shape == (n_pos_ham, cfg.batch_size) )

    # For each pair of corresponding cells, one should be 0.
    my_assert (lambda : not np.any(NeighLoss * NonNeighLoss))
    Loss = NeighLoss + NonNeighLoss
    my_assert (lambda : (all(Loss.flat >= 0.0) and all(Loss.flat <= 1.0)))
    end(t)
    return Loss

def calc_IsPossDistLteActual(dists_vec, cfg):
    """
    Each column has Trues @top, Falses @bottom.
    Bottom-most True: its row corresponds to num bits flipped.
    """
    # Every col is 1:n_bits.
    col = np.arange(1, cfg.n_bits+1).reshape(cfg.n_bits, 1)
    PossibleDistances = repmat(col, 1, cfg.batch_size)
    my_assert (lambda : PossibleDistances.shape == cfg.shape)

    # Every row is dists_vec (0 <= x <= n_bits).
    ActualDistances = repmat(dists_vec, cfg.n_bits, 1)
    my_assert (lambda : ActualDistances.shape == cfg.shape)

    Bools = PossibleDistances <= ActualDistances
    my_assert (lambda : Bools.shape == cfg.shape)
    return Bools

def uniq_SortedDeltaIdxs(SortedDeltaIdxs, cfg):
    """
    What is the significance of this modification?
    Each col of SortedDeltaIdxs had been the nums 0..n_bits, variously sorted.
    Add to each value some multiple of n_bits.
    Makes each cell unique.
    """
    # x: numbers grow by n_bits w/ each column.
    row = np.arange(0, cfg.batch_size * cfg.n_bits, cfg.n_bits)
    my_assert (lambda : len(row) == cfg.batch_size)
    ColMask = repmat(row, cfg.n_bits, 1)
    # All vals in col are same.
    my_assert (lambda : ColMask.shape == cfg.shape)
    my_assert (lambda : SortedDeltaIdxs.shape == cfg.shape)
    SortedDeltaIdxs += ColMask
    # print 'UniqSortedDeltaIdxs';  print SortedDeltaIdxs
    my_assert (lambda : all(np.unique(SortedDeltaIdxs.flat) == np.arange(cfg.n_bits * cfg.batch_size)))
    return SortedDeltaIdxs

def get_linear_idxs_of_diff_bits(SortedDeltaIdxs, dists_vec, cfg):

    # Identifies spots in matrix where the indices of diff bits will be.
    IsPossDistLteActual = calc_IsPossDistLteActual(dists_vec, cfg)

    # Columnar linear indices for the whole mtx.
    UniqSortedDeltaIdxs = uniq_SortedDeltaIdxs(SortedDeltaIdxs, cfg)
    
    # >> Which bits were actually different!  (Pts were on opposite sides of HP.) <<
    #
    # Hack:
    # We +1 to the idxs, so that none has the value 0.
    # Then we replace all the idxs we don't want with 0.
    # For those left, we shift back down one.
    LinearIdxsOfDiffBits = (UniqSortedDeltaIdxs + 1) * IsPossDistLteActual
    my_assert (lambda : LinearIdxsOfDiffBits.shape == cfg.shape)
    # print 'LinearIdxsOfDiffBits (+1)';  print LinearIdxsOfDiffBits

    diff_bits = np.extract(LinearIdxsOfDiffBits > 0, LinearIdxsOfDiffBits)
    diff_bits -= 1
    # --------------------------------
    # extract(): returns non-zero vals by going along ROWs,
    # not down COLs (as in Matlab).  So the ORDER of these may be wacky.
    # Sorting gives us the same order that Matlab would.
    # --------------------------------
    diff_bits.sort()

    my_assert (lambda :
               len(diff_bits) ==
               len(np.extract(IsPossDistLteActual == True, IsPossDistLteActual)))
    
    print 'diff_bits:', diff_bits
    return diff_bits

def mk_Y1p_Y2p(Y1, Y2, dists_vec, SortedDeltaIdxs, SumIdxs, DiffIdxs, cfg):
    """
    (y1p, y2p) are the solutions to loss-adjusted inference ???
    """
    # -- Each column represents one case. --
    t = beg('Y1p, Y2p')

    # The indices go DOWN COLUMNS, starting at 0.
    # (In NumPy, normally they go the other way, across rows.)

    # Y1p, Y2p:
    #   When bits were SAME (between x1, x2):
    #     both:
    #       -1: average was negative
    #       +1: average was positive
    #   When bits were DIFF:
    #     Y1p:
    #       -1: bits diff, x2 was larger
    #       +1: bits diff, x1 was larger
    #     Y2p (opposite):
    #       -1: bits diff, x1 was larger
    #       +1: bits diff, x2 was larger

    # -- Set when bits were DIFF. --
    
    lindxs_diff_bits = get_linear_idxs_of_diff_bits(SortedDeltaIdxs, dists_vec, cfg)
    my_assert (lambda : len(lindxs_diff_bits) <= len(SortedDeltaIdxs.flatten()))
    vals = mtx_vals(DiffIdxs, lindxs_diff_bits)
    vals = 2 * (1-vals) - 1
    Y1p = mtx_assigns(np.zeros(cfg.shape).astype(int),
                      lindxs_diff_bits,
                      vals)
    Y2p = -Y1p
    print 'Y1p';  print Y1p
    print 'Y2p';  print Y2p
    def are_all_signs(M):
        vals = M.flatten()
        return all(vals >= -1) and all(vals <= 1)
    my_assert (lambda : are_all_signs(Y1p))
    my_assert (lambda : are_all_signs(Y2p))

    # -- Set when bits were SAME. --

    # transpose(): to get indices DOWN COLS.
    lindxs_same_bits = find(Y1p.transpose() == 0)
    print 'lindxs_same_bits';  print lindxs_same_bits
    # No overlap.
    my_assert (lambda : set(lindxs_same_bits).isdisjoint(set(lindxs_diff_bits)))
    
    vals = mtx_vals(SumIdxs, lindxs_same_bits)
    vals = 2 * (1-vals) - 1
    def set_sames(Mtx):
        return mtx_assigns(Mtx,
                           lindxs_same_bits,
                           vals)
    Y1p = set_sames(Y1p)
    Y2p = set_sames(Y2p)
    def all_are_unit(M):
        vals = M.flatten()
        return all((vals == 1) + (vals == -1))
    my_assert (lambda : all_are_unit(Y1p))
    my_assert (lambda : all_are_unit(Y2p))
    end(t)
    return Y1p, Y2p

#------------------------------------

def select(cfg):
    """
    Standard Matrix shape: (n_bits, batch_size)
    Each col: one case.
    Each row, either:
      - Hamming distance
      - bit of sketch
    """
    # PDF: "Principles of Hash-based Text Retrieval".
    # Hash-based search methods operationalize — apparently or hidden
    # — a means for *embedding* high-dimensional vectors into a
    # low-dimensional space.
    # Employ a family (fancy-H_psi) of simple hash functions.
    
    # x: the original high-dimensional vectors.
    
    # x1, x2: np.arrays of FVs.
    # To make the FVs into col vectors, need to center (mk_center_x).
    x1, x2, is_near_v = get_fvs(cfg)
   
    Loss = mk_Loss(cfg, is_near_v)
    del is_near_v

    W = get_W(cfg)

    # Wx1, Wx2 are "low-dimensional embeddings".
    # 'Embedding': representation of something in a particular space.
    # Reduced the number of dimensions from p -> q.
    # (But we haven't yet quantized.)
    Wx1, Wx2 = get_Wx1_Wx2(W, x1, x2, cfg)

    # For dot-products between HPs and FVs.
    # 'Delta' -- between the abs(diff) and the abs(sum).
    # SumIdxs:  {0,1}:  Average is... (0) positive, or (1) negative.
    # DiffIdxs: {0,1}   Larger?...    (0) x1,       or (1) x2.
    SortedDeltas, SortedDeltaIdxs, SumIdxs, DiffIdxs = calc_dot_prods(Wx1, Wx2, cfg)

    dists_vec = get_dists_vec(SortedDeltas, Loss, cfg)
    del SortedDeltas

    # Y: "Quantized low-dimensional embeddings" (i.e. the encodings or 'sketches)'.
    # sign(): means of quantization (in this case, binarization).
    Y1 = np.sign(Wx1).astype(int)
    Y2 = np.sign(Wx2).astype(int)
    
    # What's "p" for?  "Prime"?
    Y1p, Y2p = mk_Y1p_Y2p(Y1, Y2, dists_vec, SortedDeltaIdxs, SumIdxs, DiffIdxs, cfg)
    my_assert (lambda : Y1.shape == cfg.shape)

    # nonzero_grad_1:
    #    one boolean per column.  (true if sum of column is non-zero.)
    #    ('sum' takes the sum OF EACH COLUMN.)

    print 'Y1';   print Y1
    print 'Y1p';  print Y1p
    print '(Y1 - Y1p)';  print (Y1 - Y1p)

    print 'Y2';   print Y2
    print 'Y2p';  print Y2p
    print '(Y2 - Y2p)';  print (Y2 - Y2p)

    Ydiff1 = Y1 - Y1p
    Ydiff2 = Y2 - Y2p

    def nonzero_grad(M): return np.sum(abs(M), axis=0) != 0
    nonzero_grad_1 = nonzero_grad(Ydiff1)
    nonzero_grad_2 = nonzero_grad(Ydiff2)

    print nonzero_grad_1
    print nonzero_grad_2

    print Y1p.shape
    print nonzero_grad_1.shape


    print x1[:, nonzero_grad_1]
    print Y1[:, nonzero_grad_1]
    print Y1p[:,nonzero_grad_1]

    print (Y1[:, nonzero_grad_1] - Y1p[:,nonzero_grad_1]).transpose()
    print (Y2[:, nonzero_grad_2] - Y2p[:,nonzero_grad_2]).transpose()

    # -- Compute the gradient --

    # grad = [x1(:,nonzero_grad_1) * (y1(:,nonzero_grad_1) - y1p(:,nonzero_grad_1))' + ...
    #         x2(:,nonzero_grad_2) * (y2(:,nonzero_grad_2) - y2p(:,nonzero_grad_2))']';

    a = x1[:,nonzero_grad_1]
    b = Y1[ :,nonzero_grad_1]
    c = Y1p[:,nonzero_grad_1]
    d = x2[:,nonzero_grad_2]
    e = Y2[ :,nonzero_grad_2]
    f = Y2p[:,nonzero_grad_2]

    # a * (b - c).transpose() + d * (e - f).transpose()

    grad = [

        # x1: nparray of FVs.
        # Select the FVs to keep; turn into centered nparrays (mk_center_x).
        # The resulting mtx will have shape: (p, len(nonzero_grad_1)).
        # Y1 is of shape: cfg.shape.

        a * (b - c).transpose() + d * (e - f).transpose()

        # x1[:,nonzero_grad_1] * (
        #     Y1[ :,nonzero_grad_1] -
        #     Y1p[:,nonzero_grad_1]
        #     ).transpose() +
        
        # X2[:,nonzero_grad_2] * (
        #     Y2[ :,nonzero_grad_2] -
        #     Y2p[:,nonzero_grad_2]
        #     ).transpose()
        
        ].transpose()

    #
    # We're gathering up columns.
    # Cols from X1, multiplied by Ydiff1 cols transposed.
    # Cols from X2, multiplied by Ydiff2 cols transposed.
    #
    # Add the mtxs, then transpose the result.
    # 


if __name__ == '__main__':
    train_cfg = get_params()
    my_assert (lambda : train_cfg.rho <= train_cfg.n_bits)
    select(train_cfg)
