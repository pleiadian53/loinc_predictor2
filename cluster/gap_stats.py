# encoding: utf-8

import time, os, re, collections, sys, random 
import hashlib
import scipy

import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

import learn_manifold

plt.rcParams['figure.figsize'] = 10, 10


def gap_stats(**kargs):
    return eval(**kargs) 
def eval(data, refs=None, nrefs=20, ks=range(2,100), n_init=10):
    """
    Compute the Gap statistic for an nxm dataset in data using gapkmean (from standard Python library)

    Either give a precomputed set of reference distributions in refs as an (n,m,k) scipy array,
    or state the number k of reference distributions in nrefs for automatic generation with a
    uniformed distribution within the bounding box of data.

    Give the list of k-values for which you want to compute the statistic in ks.

    Reference
    ---------
    https://pypi.python.org/pypi/gapkmean/1.0

    """
    from gap import gap 
    gaps, s_k, K = gap.gap_statistic(data, refs=refs, B=nrefs, K=ks, N_init=n_init)
    opt_k = gap.find_optimal_k(gaps, s_k, K)

    return opt_k

def gap_stats0(data, refs=None, nrefs=20, ks=range(1,20)):
    """
    Compute the Gap statistic for an nxm dataset in data.

    Either give a precomputed set of reference distributions in refs as an (n,m,k) scipy array,
    or state the number k of reference distributions in nrefs for automatic generation with a
    uniformed distribution within the bounding box of data.

    Give the list of k-values for which you want to compute the statistic in ks.

    Reference
    ---------
    https://gist.github.com/michiexile/5635273#file-gap-py

    """
    import scipy
    import scipy.cluster.vq
    import scipy.spatial.distance
    dst = scipy.spatial.distance.euclidean

    shape = data.shape
    if refs==None:
        tops = data.max(axis=0)
        bots = data.min(axis=0)
        dists = scipy.matrix(scipy.diag(tops-bots))
        rands = scipy.random.random_sample(size=(shape[0],shape[1],nrefs))
        for i in range(nrefs):
            rands[:,:,i] = rands[:,:,i]*dists+bots
    else:
        rands = refs

    gaps = scipy.zeros((len(ks),))
    for (i,k) in enumerate(ks):
        (kmc,kml) = scipy.cluster.vq.kmeans2(data, k)
        disp = sum([dst(data[m,:],kmc[kml[m],:]) for m in range(shape[0])])
        refdisps = scipy.zeros((rands.shape[2],))
        for j in range(rands.shape[2]):
            (kmc,kml) = scipy.cluster.vq.kmeans2(rands[:,:,j], k)
            refdisps[j] = sum([dst(rands[m,:,j],kmc[kml[m],:]) for m in range(shape[0])])
        # gaps[i] = scipy.log(scipy.mean(refdisps))-scipy.log(disp)
        gaps[i] = scipy.mean(scipy.log(refdisps))-scipy.log(disp)
    return gaps

def generate(**kargs): 
    # from sklearn.datasets.samples_generator import make_blobs
    x, y = make_blobs(750, n_features=2, centers=12)

    plt.scatter(x[:, 0], x[:, 1])
    plt.show()

    return 

def optimalK(data, nrefs=15, maxClusters=20, **kargs):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)

    Reference
    ---------
    https://anaconda.org/milesgranger/gap-statistic/notebook
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})

    step = kargs.get('step', 1)  # steps in cluster numbers
    minClusters = kargs.get('minClusters', 1)

    for gap_index, k in enumerate(range(minClusters, maxClusters, step)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)
        
        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

    return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal

def plot_gap_stats(gapdf, opt_k, **kargs): 

    identifier = kargs.get('identifier', 'gap_analysis')
    outputdir = kargs.get('outputdir', os.path.join(os.getcwd(), 'plot'))
    if not os.path.exists(outputdir): os.makedirs(outputdir) # base directory

    k = opt_k
    plt.clf()
    plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
    plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
    plt.grid(True)
    plt.xlabel('Cluster Count')
    plt.ylabel('Gap Value')
    plt.title('Gap Values by Cluster Count')
    # plt.show()

    graph_ext = 'tif'
    fpath = os.path.join(outputdir, 'gap_stats-%s.%s' % (identifier, graph_ext))
    print('output> saving silhouette test result to %s' % fpath)
    plt.savefig(fpath)
     
    return 

def plot_cluster(X, opt_k, **kargs):  # limit 2D
    identifier = kargs.get('identifier', 'opt_k')
    outputdir = kargs.get('outputdir', os.path.join(os.getcwd(), 'plot'))
    if not os.path.exists(outputdir): os.makedirs(outputdir) # base directory

    k = opt_k 
    km = KMeans(k)
    km.fit(X)

    df = pd.DataFrame(X, columns=['x','y'])
    df['label'] = km.labels_

    colors = plt.cm.Spectral(np.linspace(0, 1, len(df.label.unique())))

    for color, label in zip(colors, df.label.unique()):
        tempdf = df[df.label == label]
        plt.scatter(tempdf.x, tempdf.y, c=color)
    
    plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], c='r', s=500, alpha=0.7, )
    plt.grid(True)
    # plt.show()

    graph_ext = 'tif'
    fpath = os.path.join(outputdir, 'gap_stats-%s.%s' % (identifier, graph_ext))
    print('output> saving silhouette test result to %s' % fpath)
    plt.savefig(fpath)

    return 

def t_gap(**kargs): 
    X, y = make_blobs(750, n_features=2, centers=12)
    k, gapdf = optimalK(X, nrefs=15, maxClusters=25)
    print 'Optimal k is: ', k
  
    plot_gap_stats(gapdf, k)
    plot_cluster(X, k)

    return 

def find_optimal_k(X, y=None, **kargs): 
    """
    Input
    -----
    X
    y: probably not necessary 
    identifier: 
    reduce_dimension: 

    """
    # import learn_manifold
    default_id = 'nKgap_stats'
    identifier = kargs.get('identifier', default_id)

    X_proj = X
    dim0 = X.shape[1]
    if kargs.get('reduce_dimension', False): # dimensionality reduction prior to gap statistical analysis
        X_proj = learn_manifold.tsne(X, identifier=identifier)  # use t-SNE by default
        print('optimal_k> dim of X from %d to %d' % (dim0, X_proj.shape[1]))
        assert X_proj.shape[0] == X.shape[0]
    
    step = kargs.get('step', 1)
    min_n_clusters, max_n_clusters = kargs.get('min_n_clusters', 1), kargs.get('max_n_clusters', 20)
    nrefs = kargs.get('nrefs', 15)

    k, gapdf = optimalK(X_proj, nrefs=nrefs, minClusters=min_n_clusters, maxClusters=max_n_clusters, step=step)
    print 'result> Optimal k is: ', k

    plot_gap_stats(gapdf, opt_k=k, identifier=identifier)

    return k

def t_gap_sprint(**kargs): 
    from seqmaker import evaluate
    import sprint_config
    # import cluster_config

    prefix = os.path.join(kargs.get('prefix', os.getcwd()), 'data')
    fpath = os.path.join(prefix, 'tset-Esprint-Tbaseline.csv')
    ts = pd.read_csv(fpath, sep=',', header=0, index_col=False, error_bad_lines=True)
    print('io> loaded data of dim %s from:\n%s\n' % (str(ts.shape), fpath)) # [log] dim: (8808, 16)

    meta_fields = sprint_config.TSet.meta_fields
    X, y = evaluate.transform(ts, standardize_='minmax', meta_fields=meta_fields)
    
    k, gapdf = optimalK(X, nrefs=15, maxClusters=100)
    print 'Optimal k is: ', k

    plot_gap_stats(gapdf, opt_k=k, identifier='sprint_gap_analysis')

    # limit: 2D plot
    # plot_cluster(X, opt_k=k, identifier='sprint_opt_k')

    return 

def t_gap_sprint2(**kargs): 
    from seqmaker import evaluate
    import sprint_config
    import cluster
    import bisect

    ## load data set (e.g. SPRINT data under sprint/data)
    prefix = os.path.join('/phi/proj/poc7002/tpheno/sprint', 'data') # os.path.join(kargs.get('prefix', os.getcwd()), 'data-exp')
    fpath = os.path.join(prefix, 'tset-Esprint-Tbaseline.csv')
    ts = pd.read_csv(fpath, sep=',', header=0, index_col=False, error_bad_lines=True)
    print('io> loaded data of dim %s from:\n%s\n' % (str(ts.shape), fpath)) # [log] dim: (8808, 16)

    meta_fields = sprint_config.TSet.meta_fields
    X, y = evaluate.transform(ts, standardize_='minmax', meta_fields=meta_fields)

    # [log] 22 seems to be the best
    # bestK = eval(X, refs=None, nrefs=20, ks=range(2,100))
    # print('result> best k: %d' % bestK)  

    # [params]
    step = 1
    min_n_clusters, max_n_clusters = 20, 250
    nrefs = 15
    print('params> max_n_clusters: %d (step=%d), nrefs: %d' % (max_n_clusters, step, nrefs))

    bestK0 = find_optimal_k(X, y=y, reduce_dimension=True, identifier='sprint_Rtsne', 
        step=step, min_n_clusters=min_n_clusters, max_n_clusters=max_n_clusters)
    print('result> best k with dim reduction (via t-SNE): %d' % bestK0) # [log] 161 (40-200)
    bestK = find_optimal_k(X, y=y, reduce_dimension=False, identifier='sprint_Rnoop', 
        step=step, min_n_clusters=min_n_clusters, max_n_clusters=max_n_clusters)
    print('result> best k WITHOUT dim reduction: %d' % bestK)  # [log] 159 (40-200)
     
    # bestK = 160 # 40-200
    ### compare to silhouette method ### 
    range_n_clusters = range(max(bestK-10, 1), min(bestK+10, X.shape[0]))
    for candidate_k in (2, 5, 10, 20, 50, 100):
        if not candidate_k in range_n_clusters: 
            bisect.insort_left(range_n_clusters, candidate_k)

    print('params> range_n_clusters:\n%s\n' % range_n_clusters)
    bestK2 = cluster.run_silhouette_analysis(X=X, y=y, range_n_clusters=range_n_clusters, reduce_dimension=True, identifier='sprint_Rtsne')
    print('result> best k via silhouette scores + dim reduction (via t-SNE): %d' % bestK2)
    bestK2 = cluster.run_silhouette_analysis(X=X, y=y, range_n_clusters=range_n_clusters, reduce_dimension=False, identifier='sprint_Rnoop')
    print('result> best k via silhouette scores WITHOUT dim reduction: %d' % bestK2)

    return

def test(**kargs): 
    # t_gap(**kargs)
    # t_gap_sprint()
    t_gap_sprint2()

    return 

if __name__ == "__main__": 
    test()

