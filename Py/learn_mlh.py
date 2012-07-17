#!/usr/bin/env python

from collections import namedtuple

train_set = [1,3,5]

# ParamSet = namedtuple('ParamSet',
#                       ['n_bits',
#                        'batch_size'])

class ParamSet(object):
    n_bits       = 64   # num bits (q)
    # gradient descent
    batch_size   = 100     # for gradient descent
    max_iter     = 500
    zero_bias_p  = True
    eta          = 0.1     # learning rate
    shrink_eta_p = True
    momentum     = 0.9
    # validation
    n_val_during = 5
    n_val_after  = 1
    shrink_w_set = []   # range of values
    # loss hyperparams
    rho          = 4
    lamduh       = 0.5
    # training data
    mode         = 'training'    # or 'test', ???
    train_set    = train_set
    num_train    = len(train_set)

# standard-normal distro.
#   #rows: n_bits
#   #cols: n_input_features
#          then add 1 more col (of 0s)
#
def create_init_W(n_rows, n_cols)
    def one_row_W(n_cols):
        xs = [0.1 * random.gauss(0, 1) for i in range(n_cols)]
        xs.append(0.0)
        return xs
    return [one_row_W(n_cols) for i in range(n_rows)]

def learn_mlh(data, params, init_W):
    pass

"""
Winc:
  - matrix the size of W
  - running average of gradient???
  - initialized as 0s
  - * momentum (0 <-> 1) -- that shrinks it
  - + (eta * ???) -- see eta below
  -   

eta: learning rate
  - starts at 0.1
  - may shrink over time in ever-increasing increments
  - learn faster at the start, learn more slowly later
  - used to calculate (updated) Winc

caseLinearIdxs:
  - row vector  size:(nCases, 1)
  - reused with each epoch
  - the indices of every instance in the batch
  - indices go DOWN columns

neighOrNot:
  - column vector
  - booleans
  - corresponding to each training pair

x1, x2:
  - matrix
  - each column is a training point
  - w/ an addition 1 (at end of each tr. pt.)
  - x1: 1st pt in labeled pair
  - x2: 2nd pt in labeled pair

"""
