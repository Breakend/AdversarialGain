import numpy as np
import scipy as sp

import scipy.stats

EPS = 1e-8

def cosine(u, v):
    return max((1.0 - (np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))), 0.0)

def jsd(p, q, base=np.e):
    '''
        Implementation of pairwise `jsd` based on
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    '''
    p = np.asarray(p)
    q = np.asarray(q)
    m = 1./2*(p + q)
    return sp.stats.entropy(p,m, base=base)/2. +  sp.stats.entropy(q, m, base=base)/2.


def normalize_array(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
