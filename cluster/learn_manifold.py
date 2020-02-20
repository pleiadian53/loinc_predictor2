# encoding: utf-8

# print(__doc__)
import sys, os, random 

from pandas import DataFrame 
import pandas as pd
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

### plotting (1)
import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt

from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import matplotlib.patheffects as PathEffects

from sklearn import manifold, datasets

### local modules ### 
try: 
    from utils import div
except: 
    from utils_sys import div

# from seqmaker.seqparams import name_image_file  # cannot import if run within seqmaker 

from cluster_config import name_image_file


### plotting (2)
import itertools
import matplotlib.cm as cm

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

# Next line to silence pyflakes. This import is needed.
Axes3D

TestDir = os.path.join(os.getcwd(), 'plot')
if not os.path.exists(TestDir): os.makedirs(TestDir) # test directory

def make_s_curve(**kargs): 
    n_points = 1000
    X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0) # [log] dim > X: (1000, 3), color: (1000,)
    
    n_neighbors = 10
    n_components = 2

    outputdir = kargs.get('outputdir', os.path.join(os.getcwd(), 'plot'))
    if not os.path.exists(outputdir): os.makedirs(outputdir) # base directory

    fig = plt.figure(figsize=(15, 8))
    plt.suptitle("Manifold Learning with %i points, %i neighbors"
                 % (1000, n_neighbors), fontsize=14)

    try:
        # compatibility matplotlib < 1.0
        ax = fig.add_subplot(251, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
        ax.view_init(4, -72)
    except:
        ax = fig.add_subplot(251, projection='3d')
        plt.scatter(X[:, 0], X[:, 2], c=color, cmap=plt.cm.Spectral)

    ext = 'tif'
    fpath = os.path.join(TestDir, name_image_file(descriptor='scurve', **kargs))
    plt.savefig(fpath, dpi=300)
    plt.close()  

    return (X, color)

def locally_linear(X, **kargs): 
    # import matplotlib.cm as cm

    n_neighbors = 10
    n_components = 2

    y = kargs.get('y', None)

    methods = ['standard', 'ltsa', 'hessian', 'modified']
    labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']
    
    plt.clf()
    fig = plt.figure(figsize=(15, 8))

    n_points = X.shape[0]

    outputdir = kargs.get('outputdir', os.path.join(os.getcwd(), 'plot'))
    if not os.path.exists(outputdir): os.makedirs(outputdir) # base directory

    # configure color(s) 
    colors = itertools.cycle(cm.rainbow(np.linspace(0, 1, n_points)))

    for i, method in enumerate(methods):
        t0 = time()
        Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                        eigen_solver='auto',
                                        method=method).fit_transform(X)
        t1 = time()
        print("%s: %.2g sec" % (methods[i], t1 - t0))

        ax = fig.add_subplot(252 + i)
        plt.scatter(Y[:, 0], Y[:, 1], c=next(colors), cmap=plt.cm.Spectral)
        plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')
    
        # scatter2(X_proj, y)
        fpath = os.path.join(TestDir, name_image_file(descriptor=method, **kargs)) # kargs: seq_ptype, d2v_method
        plt.savefig(fpath, dpi=300)
        plt.close()
    return 

def isomap(X, **kargs): 
    # import matplotlib.cm as cm

    n_neighbors = 10
    n_components = 2

    y = kargs.get('y', None)
    # ulabels = labels = y = kargs.get('y', None)
    # n_labels = X.shape[0]
    # if labels is not None: 
    #     ulabels = set(labels)
    #     n_labels = len(ulabels)

    t0 = time()
    X_proj = Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
    t1 = time()
    print("Isomap: %.2g sec" % (t1 - t0))

    outputdir = kargs.get('outputdir', os.path.join(os.getcwd(), 'plot'))
    if not os.path.exists(outputdir): os.makedirs(outputdir) # base directory

    plot_mode = kargs.get('graph_mode', 'seaborn')
    fpath = os.path.join(TestDir, name_image_file(descriptor='isomap', **kargs)) # kargs: seq_ptype, d2v_method

    if plot_mode.startswith('s'):
        scatter2(X_proj, y)
        plt.savefig(fpath, dpi=300)
    else: 
        # [params] colors 
        n_points = X.shape[0]
        colors = itertools.cycle(cm.rainbow(np.linspace(0, 1, n_points)))

        # for i, c in enumerate(itertools.cycle(cm.rainbow(np.linspace(0, 1, n_points)))): 
        # for i, c in enumerate(itertools.cycle(["r", "b", "g"])): 
        lcmap = {0: 'g', 1: 'b', 2: 'r', }
        colors = (lcmap[l] for l in y) # loop through y 

        fig = plt.figure(figsize=(15, 8))

        ax = fig.add_subplot(257)
        plt.scatter(X_proj[:, 0], X_proj[:, 1], c=next(colors), cmap=plt.cm.Spectral)  # 
        plt.title("Isomap (%.2g sec)" % (t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')
        plt.savefig(fpath, dpi=300)

    plt.close()

    return X_proj 

def mds(X, **kargs):
    # import matplotlib.cm as cm

    # n_neighbors = 10
    n_components = 2
    y = kargs.get('y', None)

    t0 = time()
    mds = manifold.MDS(n_components, max_iter=100, n_init=1)
    X_proj = Y = mds.fit_transform(X)
    t1 = time()
    print("MDS: %.2g sec" % (t1 - t0))

    outputdir = kargs.get('outputdir', os.path.join(os.getcwd(), 'plot'))
    if not os.path.exists(outputdir): os.makedirs(outputdir) # base directory

    plot_mode = kargs.get('graph_mode', 'seaborn')
    fpath = os.path.join(TestDir, name_image_file(descriptor='mds', **kargs)) # kargs: seq_ptype, d2v_method
 
    if plot_mode.startswith('s'):
        scatter2(X_proj, y)
        plt.savefig(fpath, dpi=300)
    else: 

        # [params] colors 
        n_points = X_proj.shape[0]
        # colors = itertools.cycle(cm.rainbow(np.linspace(0, 1, n_points)))

        lcmap = {0: 'g', 1: 'b', 2: 'r', }
        colors = (lcmap[l] for l in y) # loop through y 

        fig = plt.figure(figsize=(15, 8))

        ax = fig.add_subplot(258)
        plt.scatter(X_proj[:, 0], X_proj[:, 1], c=next(colors), cmap=plt.cm.Spectral)
        plt.title("MDS (%.2g sec)" % (t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight') 
        plt.savefig(fpath, dpi=300)

    plt.close()

    return Y

def spectral(X, **kargs):
    n_neighbors = 10
    n_components = 2

    y = kargs.get('y', None)

    t0 = time()
    se = manifold.SpectralEmbedding(n_components=n_components,
                                n_neighbors=n_neighbors)
    X_proj = Y = se.fit_transform(X)
    t1 = time()
    print("SpectralEmbedding: %.2g sec" % (t1 - t0))

    outputdir = kargs.get('outputdir', os.path.join(os.getcwd(), 'plot'))
    if not os.path.exists(outputdir): os.makedirs(outputdir) # base directory

    plot_mode = kargs.get('graph_mode', 'seaborn')
    fpath = os.path.join(TestDir, name_image_file(descriptor='spectral', **kargs)) # kargs: seq_ptype, d2v_method

    if plot_mode.startswith('s'):
        scatter2(X_proj, y)
        plt.savefig(fpath, dpi=300)
    else: 
        # [params] colors 
        n_points = X_proj.shape[0]
        # colors = itertools.cycle(cm.rainbow(np.linspace(0, 1, n_points)))

        lcmap = {0: 'g', 1: 'b', 2: 'r', }
        colors = (lcmap[l] for l in y) # loop through y 

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(259)
        plt.scatter(X_proj[:, 0], X_proj[:, 1], c=next(colors), cmap=plt.cm.Spectral)
        plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight') 
        plt.savefig(fpath, dpi=300)

    plt.close()

    return Y

def tsne(X, **kargs): 
    import collections
    # n_neighbors = 10
    n_components = 2
    y = kargs.get('y', None)

    if y is not None: 
        lcounts = collections.Counter(y)
        n_labels = len(set(y))
        print('verify> number of unique labels: %d > %s' % (n_labels, lcounts))
    else: 
        y = np.repeat(1, X.shape[0])

    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    X_proj = tsne.fit_transform(X)
    print('output> dim | X_proj: %s' % str(X_proj.shape))
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))

    plot_mode = kargs.get('graph_mode', 'seaborn')
    identifier = kargs.get('identifier', 'sprint')

    outputdir = kargs.get('outputdir', os.path.join(os.getcwd(), 'plot'))
    if not os.path.exists(outputdir): os.makedirs(outputdir) # base directory
    # [output]
    fpath = os.path.join(outputdir, 'tsne-%s.tif' % identifier) # kargs: seq_ptype, d2v_method

    # plot_embedding(X_proj, y, title='t-SNE (D2V: %s, Seq: %s)' % (kargs.get('d2v_method', '?'), kargs.get('seq_ptype', '?')))
    # plt.savefig(fpath, dpi=300)
    # plt.clf()
    
    if plot_mode.startswith('s'):
        fpath = os.path.join(outputdir, 'tsne-seaborn-%s.tif' % identifier) # kargs: seq_ptype, d2v_method
        scatter2(X_proj, y)
        plt.savefig(fpath, dpi=300)
        
    else:    

        # [params] colors 
        n_points = X_proj.shape[0]
        # colors = itertools.cycle(cm.rainbow(np.linspace(0, 1, n_points))) 
        # palette = np.array(sns.color_palette("hls", n_labels)) # [m1] circular color system

        lcmap = {0: 'g', 1: 'b', 2: 'r', }
        colors = (lcmap[l] for l in y) # loop through y 

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(2, 5, 10)

        # plt.scatter(Y[:, 0], Y[:, 1], c=next(colors), cmap=plt.cm.Spectral)
        plt.scatter(X_proj[:, 0], X_proj[:, 1], c=next(colors), cmap=plt.cm.Spectral) # next(colors)

        plt.title("t-SNE (%.2g sec)" % (t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

        fpath = os.path.join(outputdir, 'tsne-%s.tif' % identifier) # kargs: seq_ptype, d2v_method
        plt.savefig(fpath, dpi=300)

    plt.close()

    return X_proj

def scatter(x, colors):
    # import seaborn as sns
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

def scatter2(x, y):
    """

    Memo
    ----
    1. circular color system: 
        http://seaborn.pydata.org/tutorial/color_palettes.html

    """
    # import seaborn as sns
    labels = y
    assert x.shape[0] == len(y)

    example_label = labels[random.sample(labels, 1)[0]]
    print('verify> label: %s' % example_label)

    ulabels = set(labels) # unique labels
    n_colors = len(ulabels)

    lcmap = {}  # label to color id: assign a color to each unique label 
    for ul in ulabels: 
        if not lcmap.has_key(ul): lcmap[ul] = len(lcmap)
    clmap = {c: l for l, c in lcmap.items()}  # reverse map: color id => label
    
    # from labels to colors
    colors = np.zeros((len(labels), ), dtype='int') # [0] * n_colors
    for i, l in enumerate(labels): 
        colors[i] = lcmap[l]  # label => color id

    palette = np.array(sns.color_palette("hls", n_colors)) # [m1] circular color system

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(n_colors): # foreach color
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0) # find all fvec of i-th color
        txt = ax.text(xtext, ytext, str(clmap[i]), fontsize=18)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts 

def plot_embedding(X, y, title=None):
    from matplotlib import offsetbox

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(y[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def test(**kargs): 
    return

if __name__ == "__main__": 
    test()
