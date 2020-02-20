# encoding: utf-8
import os, sys, re
import numpy as np 

# import matplotlib
# matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
# from matplotlib import pyplot as plt

from pandas import DataFrame, Series
import pandas as pd

###############################################################################################################
#
#   Note 
#   ---- 
#   Subsummed by sampling_utils
#   Refactored from tpheno.seqmaker.seqUtils
#
#   Related 
#   -------
#   tpheno.seqmaker.seqSampling
# 
#
###############################################################################################################

def sample_class(ts, **kargs):
    """
    Sample training data proportional to class labels. 
    """
    target_field = kargs.get('target_field', 'target')

    # [params]
    ignore_index = kargs.get('ignore_index', True)
    replace = kargs.get('replace', False)

    n_samples = ts.shape[0]
    if 'n_samples' in kargs: n_samples = kargs['n_samples']  # compatibility

    y = ts[target_field].values
    labels = list(set(y))
    n_labels = len(labels)
    # nrows = ts.shape[0]
    n_subsets = utils.divide_interval(n_samples, n_parts=n_labels) # 10 => [3, 3, 4]

    # pars = utils.partition(range(ts.shape[0]), n_labels) # partition the list into 'n_labels' parts
    pdict = {labels[i]: n_subset for i, n_subset in enumerate(n_subsets)} # label -> n_sample
    print('verify> label to n_sample pdict:\n%s\n' % pdict) # ok. [log] {0: 500, 1: 500}
    tsx = []

    for l, n in pdict.items(): 

        # Sample with or without replacement. Default = False.
        # tse = ts.loc[ts[target_field]==l]
        # if tse.shape[0] < n: 
        #     tsx.append(tse.sample(n=n, replace=True)) 
        tsx.append(ts.loc[ts[target_field]==l].sample(n=n, replace=replace))
    
    ts_sub = pd.concat(tsx, ignore_index=ignore_index) # True: reindex, False: keep original indices 
    assert ts_sub.shape[0] == n_samples

    return ts_sub

def sample_class2(X, y=[], **kargs): 
    """
    Input
    -----
    y: labels 
       use case: if labels are cluster labels (after running a clustering algorithm), then 
                 this funciton essentially performs a cluster sampling
    """
    # [params]
    # ignore_index = kargs.get('ignore_index', True)
    replace = kargs.get('replace', False)

    n_samples = X.shape[0] / 2.0
    if 'n_samples' in kargs: n_samples = kargs['n_samples']

    if y is None: 
        Xsub = sample(X, n_sample=n_samples, replace=replace)
        return (Xsub, np.array([-1] * Xsub.shape[0]))

    labels = list(set(y))
    n_labels = len(labels)
    # nrows = ts.shape[0]
    n_subsets = utils.divide_interval(n_samples, n_parts=n_labels) # 10 => [3, 3, 4]
    
    # [log] {0: 334, 1: 333, 2: 333}
    pdict = {labels[i]: n_subset for i, n_subset in enumerate(n_subsets)} # label -> n_samples


    print('verify> label to n_samples pdict:\n%s\n' % pdict) # ok. [log] {0: 500, 1: 500}
    tsx = []

    Xs, ys = [], []  # candidate indices
    for l, n in pdict.items(): 
        cond = (y == l)
        Xl = X[cond]
        # yl = y[cond]

        # sampling with replacement so 'n' can be larger than data size
        # idx = np.random.randint(Xl.shape[0], size=n)
        idx = np.random.choice(Xl.shape[0], size=n, replace=replace)
        # print('verify> select %d from %d instances' % (n, Xl.shape[0]))

        # print('verify> selected indices (size:%d):\n%s\n' % (len(idx), idx))
        Xs.append(Xl[idx, :])  # [note] numpy append: np.append(cidx, [4, 5])
        ys.append([l] * len(idx))
    
    assert len(Xs) == n_labels
    Xsub = np.vstack(Xs)  
    assert Xsub.shape[0] == n_samples  and Xsub.shape[1] == X.shape[1] 
    ysub = np.hstack(ys)
    assert ysub.shape[0] == n_samples 

    return (Xsub, ysub)

def sample(X, n_samples=-1, **kargs): 
    replace = kargs.get('replace', False)    

    if n_samples == -1: n_samples = X.shape[0] /2.0
    idx = np.random.choice(X.shape[0], size=n_samples, replace=replace)
    return X[idx, :]

def sample_class_average(ts, n_samples=1000, **kargs): 
    # todo
    return ts

def t_sampling(): 
    from sklearn import datasets

    # [note] n_classes * n_clusters_per_class must be smaller or equal 2 ** n_informative
    X, y = datasets.make_classification(n_samples=3000, n_features=20,
                                    n_informative=15, n_redundant=3, n_classes=3,
                                    random_state=42)
    n_labels = len(set(y))
    print('data> dim(X): %s, y: %s > n_labels: %d' % (str(X.shape), str(y.shape), n_labels))

    Xsub, ysub = sample_class2(X, y=y, n_samples=5000, replace=True)
    n_labels_sampled = len(set(ysub))
    print('sampled> dim(X): %s, y: %s > n_labels: %d' % (str(Xsub.shape), str(ysub.shape), n_labels_sampled))

    Xsub, ysub = sample_class2(X, n_samples=5000, replace=True)
    n_labels_sampled = len(set(ysub))
    print('sampled> dim(X): %s, y(DUMMY): %s > n_labels: %d' % (str(Xsub.shape), str(ysub.shape), n_labels_sampled))

    return

def test(): 
    t_sampling()
    return

if __name__ == "__main__": 
    test()


