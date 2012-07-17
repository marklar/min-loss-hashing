"""
Utility fns.
"""
import json, os, sys
# import heapq
import numpy as np
import time
import pymongo
from collections import namedtuple

#-------------------------------------

Cfg = namedtuple('Cfg', ['n_docs', 'n_bits', 'near_thresh', 'quantize'])

#-------------------------------------

# -- NumPy --
def mtx_vals(Mtx, lin_idxs):
    return Mtx.transpose().flatten()[lin_idxs]

def mtx_assigns(Mtx, lin_idxs, new_vals):
    """
    Using linear indices which go down columns.
    """
    r,c = Mtx.shape
    vals = Mtx.transpose().flatten()
    vals[lin_idxs] = new_vals
    return vals.reshape(c,r).transpose()

def repmat(mtx, rows, cols):
    """ Like Matlab's repmat() function. """
    return np.kron(np.ones((rows, cols)), mtx)

#-------------------------------------

def normalize_array(ary):
    return ary / sum(ary ** 2.0)

def normalize_matrix(M):
    return np.array([normalize_array(row) for row in M])

#-------------------------------------

g_assert = True
def my_assert(pred):
    global g_assert
    if g_assert:
        assert pred()
    else:
        pass

#-------------------------------------

def get_cfg():
    try:
        docs, bits, near_thresh = sys.argv[1:4]
        return Cfg(int(docs), int(bits), float(near_thresh))
    except:
        exe = sys.argv[0].lstrip('./')
        print 'Usage: {0} <NUM DOCS> <NUM BITS> <NEAR_THRESH>'.format(exe)
        sys.exit(1)

def get_docs_cfg():
    try:
        return Cfg(int(sys.argv[1]), None, None)
    except:
        exe = sys.argv[0].lstrip('./')
        print 'Usage: {0}.py <NUM DOCS>'.format(exe)
        sys.exit(1)

#-------------------------------------

g_mongo_connection = None
def mongo_connection():
    global g_mongo_connection
    if not g_mongo_connection:
        g_mongo_connection = pymongo.Connection()  # ('localhost', 27017)
    return g_mongo_connection

def mongo_coll(db_name, coll_name):
    cnx = mongo_connection()
    db = cnx[db_name]
    return db[coll_name]

# Unused.
def mongo_clean_coll(db_name, coll_name):
    coll = mongo_coll(db_name, coll_name)
    coll.drop()
    return coll

#-------------------------------------

SORT = [('magnitude', pymongo.DESCENDING)]
def n_biggest_fvs(n):
    coll = mongo_coll('mlh', 'fvs')
    coll.ensure_index(SORT)
    return coll.find().sort(SORT).limit(n)

#-------------------------------------

def groups_of(seq, n):
    iseq = iter(seq)
    while True:
        group = []
        try:
            for i in range(n):
                group.append(iseq.next())
        except StopIteration:
            if group: yield group
            break
        yield group

#-------------------------------------

def tick(t, interval):
    t += 1
    if t % interval == 0:
        sys.stdout.write('.')
        sys.stdout.flush()
    return t

def tock():
    print '.'

#-------------------------------------

# GLOBAL
beg_end_depth = 0

def beg(s):
    global beg_end_depth
    indent = '  ' * beg_end_depth
    beg_end_depth += 1
    print indent + s + '...'
    # sys.stdout.flush()
    return (s, time.time(), indent)

def end(t=None):
    global beg_end_depth
    beg_end_depth = max(0, beg_end_depth - 1)
    if t:
        desc, init_time, indent = t
        mins, secs = divmod(time.time() - init_time, 60)
        print indent + 'done ({0}).  ({1:02}:{2:06.3f})'.format(desc, int(mins), secs)
    else:
        print 'done.'

#-------------------------------------

def fst(xs): return xs[0]
def snd(xs): return xs[1]

#-------------------------------------

def mk_ascii(u_str):
    return u_str.encode('ascii', 'ignore')

#-------------------------------------

def mk_array(id2wt, n_dims):
    """ From alist [[int, float]], make sparse np.array. """
    indices, values = zip(*id2wt)
    sparse = np.zeros(n_dims)
    sparse[np.array(indices)] = np.array(values)
    return sparse

def alt_mk_array(id2wt, n_dims):
    x = np.zeros(n_dims).astype(np.float32)
    for i,w in id2wt: x[i] = w
    return x

#-------------------------------------

def earnest_gen(filename):
    for ln in file(filename, 'r'):
        yield json.loads(ln)

def earnest_write(file_or_filename, objs):
    if type(file_or_filename) == 'file':
        f = file_or_filename
    else:
        f = open(file_or_filename, 'w')
    with f as file:
        for o in objs:
            file.write(json.dumps(o))
            file.write('\n')

#-------------------------------------

def in_dir(dir_name, fn):
    """ Temp context switch.  Exec f in dir_name, then switch back. """
    # Difficult to use with generators and other I/O!
    orig_dir = os.getcwd()
    os.chdir(dir_name)
    res = fn()
    os.chdir(orig_dir)
    return res 

#-------------------------------------

def ord_pairs(pairs_a, pairs_b):
    """ Given two SORTED aLists, create pairs of corresponding vals. """
    iter_a, iter_b = iter(pairs_a), iter(pairs_b)
    try:
        (ka, va) = iter_a.next()
        (kb, vb) = iter_b.next()
        while True:
            if ka < kb:
                (ka, va) = iter_a.next()
            elif ka > kb:
                (kb, vb) = iter_b.next()
            else:
                yield (va, vb)
                (ka, va) = iter_a.next()
                (kb, vb) = iter_b.next()
    except StopIteration:
        pass

