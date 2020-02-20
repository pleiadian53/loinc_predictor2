import numpy as np

# need to install skdata via pip
from skdata.mnist.views import OfficialImageClassification  

import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt

import random, os, sys, re

# install tsne via pip
from tsne import bh_sne

import pandas as pd 
from pandas import DataFrame

# temporal sequence modules
from config import seq_maker_config, sys_config
from seqmaker import seqCluster as sc


def t_highD(**kargs): 
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer

    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    newsgroups = fetch_20newsgroups(subset="train", categories=categories)
    vectors = TfidfVectorizer().fit_transform(newsgroups.data)

    # verify 
    print('newsgroups.data type: %s' % type(newsgroups.data)) # a list
    print('verify> data dim: %d' % len(newsgroups.data))
    print('example:\n%s\n' % newsgroups.data[1])

    return

def run_tsne(X, y, **kargs):

    # [params]
    plot_result = kargs.get('plot_', True)
    basedir = kargs.get('output_dir', os.path.join(os.getcwd(), 'plot')) # sys_config.read('')
    if not os.path.exists(basedir): os.mkdir(basedir)
    meta = kargs.get('meta', 'test')

    X = np.asarray(X, dtype=np.float64)
    # y = np.asarray(y, dtype=np.float64)

    print('run_tsne> X dim: %s | X[1][:10]: %s' % (str(X.shape), X[1][:10]))

    # do SVD first? 

    # perform t-SNE embedding
    vis_data = bh_sne(X)

    # plot the result
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    # plot
    plt.clf()
    if plot_result: 
        plt.scatter(vis_x, vis_y, c=y, cmap=plt.cm.get_cmap("jet", 10))    	
        plt.colorbar(ticks=range(10))
        plt.clim(-0.5, 9.5)

        # save
        graph_ext = 'tif'

        fpath = os.path.join(basedir, 'tsne_%s.%s' % (meta, graph_ext))
        print('output> saving plot to %s' % fpath)
        plt.savefig(fpath, bbox_inches='tight') 
        plt.close()

    return

def t_tsne(**kargs): 

    # load up data
    data = OfficialImageClassification(x_dtype="float32")
    x_data = data.all_images
    y_data = data.all_labels

    print('info> X dim: %s' % str(x_data.shape))

    # convert image data to float64 matrix. float64 is need for bh_sne
    x_data = np.asarray(x_data).astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1))

    print('info> X dim (after conversion): %s' % str(x_data.shape))

    # For speed of computation, only run on a subset
    n = 20000
    x_data = x_data[:n]
    y_data = y_data[:n]
    
    run_tsne(X=x_data, y=y_data)

    return

def t_tsne_mnist(**kargs):
    import gzip, cPickle
    # from tsne import bh_sne 

    basedir = os.path.join(os.getcwd(), 'data')
    fpath = os.path.join(basedir, "mnist.pkl.gz")
    f = gzip.open(fpath, "rb")
    train, val, test = cPickle.load(f)
    f.close()

    X = np.asarray(np.vstack((train[0], val[0], test[0])), dtype=np.float64)
    print('verify> X (dim: %s) > example:\n%s\n' % (str(X.shape), X[0]))

    y = np.hstack((train[1], val[1], test[1]))

    X_2d = bh_sne(X)

    rcParams['figure.figsize'] = 20, 20

    scatter(X_2d[:, 0], X_2d[:, 1], c=y)

    return

def t_tsne2(**kargs): 
    """

    Memo
    ----
    1. d2v data set 
       total number of labels: 432000 > n_multilabels: 264632 > n_slabels: 6225


    """
    def mlabel_to_slabel(mlabels):  # or use seqparams.lsep 
    	labels = [mlabel.split(lsep)[0] for mlabel in mlabels] # take most frequent (sub-)label as the label
        print('verify> labels (dtype=%s): %s' % (type(labels), labels[:10]))
        return labels
    def v_lsep(y, n=10): 
        got_multilabel =True
        for i, e in enumerate(y):
            if i >= n: break 
            if len(e.split(lsep)) < 2: 
        		got_multilabel = False 
        		break 
        return got_multilabel

    from seqmaker import seqSampling as ss
    from seqmaker import seqCluster as sc 

    basedir = sys_config.read('DataExpRoot')  
    d2v_method = 'PVDM'
    vis_method = 'tsne'
    meta = kargs.get('meta', '%s-V%s' % (d2v_method, vis_method))

    redo_clustering = True
    n_sample = 20000  # still got memory error with plotly
    print('params> desired sample size: %d' % n_sample)

    lsep = '_'

    ### load data
    X, y, D = sc.build_data_matrix2(**kargs)
    n_unique_mlabels = len(set(y))

    # verifying the data
    print('t_condition_diag> X (dim: %s), y (dim: %s), D (dim: %s)' % (str(X.shape), str(y.shape), len(D)))  
    print('verify> example composite labels:\n%s\n' % y[:10])
    print('verify> got multiple label with lsep=%s? %s' % (lsep, v_lsep(y)))

    yp = mlabel_to_slabel(y)
    n_unique_slabels = len(set(yp))
    print('verify> total number of labels: %d > n_multilabels: %d > n_slabels: %d' % \
    	(len(y), n_unique_mlabels, n_unique_slabels))

    # sample subset 
    candidates = ss.get_representative(docs=D, n_sample=n_sample, n_doc=X.shape[0], policy='rand')
    # if redo_clustering: X, y, D = sc.build_data_matrix2()
    X_subset, y_subset = X[candidates], y[candidates]

    # convert image data to float64 matrix. float64 is need for bh_sne
    x_data = np.asarray(X_subset).astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1))
    y_data = y_subset

    run_tsne(X=x_data, y=y_data, meta=meta)    

    return

def run(ts, **kargs): 
    """

    Memo
    ----
    1. d2v data set 
       total number of labels: 432000 > n_multilabels: 264632 > n_slabels: 6225


    """
    def mlabel_to_slabel(mlabels):  # or use seqparams.lsep 
        labels = [mlabel.split(lsep)[0] for mlabel in mlabels] # take most frequent (sub-)label as the label
        print('verify> labels (dtype=%s): %s' % (type(labels), labels[:10]))
        return labels
    def v_lsep(y, n=10): 
        got_multilabel =True
        for i, e in enumerate(y):
            if i >= n: break 
            if len(e.split(lsep)) < 2: 
                got_multilabel = False 
                break 
        return got_multilabel

    from seqmaker import seqSampling as ss
    from seqmaker import seqCluster as sc 

    basedir = os.path.join(os.getcwd(), 'test')
    d2v_method = 'PVDM'
    vis_method = 'tsne'
    meta = kargs.get('meta', '%s-V%s' % (d2v_method, vis_method))

    redo_clustering = True
    n_sample = 20000  # still got memory error with plotly
    print('params> desired sample size: %d' % n_sample)

    lsep = '_'

    ### load data
    X, y, D = sc.build_data_matrix2(**kargs)
    n_unique_mlabels = len(set(y))

    # verifying the data
    print('t_condition_diag> X (dim: %s), y (dim: %s), D (dim: %s)' % (str(X.shape), str(y.shape), len(D)))  
    print('verify> example composite labels:\n%s\n' % y[:10])
    print('verify> got multiple label with lsep=%s? %s' % (lsep, v_lsep(y)))

    yp = mlabel_to_slabel(y)
    n_unique_slabels = len(set(yp))
    print('verify> total number of labels: %d > n_multilabels: %d > n_slabels: %d' % \
        (len(y), n_unique_mlabels, n_unique_slabels))

    # sample subset 
    candidates = ss.get_representative(docs=D, n_sample=n_sample, n_doc=X.shape[0], policy='rand')
    # if redo_clustering: X, y, D = sc.build_data_matrix2()
    X_subset, y_subset = X[candidates], y[candidates]

    # convert image data to float64 matrix. float64 is need for bh_sne
    x_data = np.asarray(X_subset).astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1))
    y_data = y_subset

    run_tsne(X=x_data, y=y_data, meta=meta)    

    return
    

def test(**kargs):
    # t_tsne(**kargs)

    # high dimension data example 
    t_highD(**kargs)

    t_tsne2(**kargs)

    return 

if __name__ == "__main__": 
    test()


