# encoding: utf-8

import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import random
import numpy as np
from pandas import DataFrame

#####################################################
#
#  Reference 
#  ---------
#  https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
#

### Main Use Case: CF-based ensemble learning and clustering ### 

def evalSimilarity(A, epsilon=1e-9):
    """
    Compute pairwise similarity between "rows" of A (i.e. assuming A is in row vector format). 

    Memo
    ----
    1. If dot product preceeds normalization, the resulting similarity matrix is not symmetric
       and cannot be used for hierarchical clustering

    """
    from sklearn.preprocessing import normalize

    A = normalize(A, axis=1, norm='l2')

    # Not recommmended => Memo [1]
    # sim = np.dot(A, A.T) # A.dot(A.T) # + epsilon
    # norms = np.array([np.sqrt(np.diagonal(sim))])
    # return (sim / norms / norms.T) 

    return np.dot(A, A.T)
### alias 
evalSimilarityByLatentFeatures = evalSimilarity  # used in {cf, utils_cf}

def toAffinity(A, sim_func=None, sig=0.5, verify=False):
    if sim_func is None: sim_func = evalSimilarity  # similarity falls in [0, 1]

    S = sim_func(A)
    if verify: 
        ep = 1e-9
        low, high = np.min(S), np.max(S)
        assert abs(low-0.0) <= ep 
        assert abs(high-1.0) <= ep

    # to distance
    S = 1. - S

    # now to similarity measure that falls within [0, 1]
    S = np.exp(- S ** 2 / (2. * sig ** 2))

    return S  # if A is symmetric, then S is symmetric

def factors_to_distance_matrix(X, var_prefix='x'): 
    # X: either a dataframe (in column vector format) or a 2D array (in row vector format) 

    # A = X.T.values if isinstance(X, DataFrame) else X 
    # ... assuming that the input dataframe is in column vector form
    cols = []
    if isinstance(X, DataFrame): 
        A = X.T.values   # assuming that X is in column vector format
        cols = X.columns.values
    else: 
        A = X

    # A: in row vector format (each instance is represented by a row vector)
    S = evalSimilarity(A)
    # print('... dim(S): {0}'.format(S.shape))
    D = 1. - S

    # turn into dataframes 
    if len(cols) == 0: cols = ['{0}_{1}'.format(var_prefix, i) for i in D.shape[0]]
    DX = DataFrame(D, columns=cols, index=cols)

    return DX
########################################################################

### Main Use Case: Show cluster structures in terms of a heatmap ###
def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z
            
        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))

def compute_serial_matrix(dist_mat, method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)
        
        compute_serial_matrix transforms a distance matrix into 
        a sorted distance matrix according to the order implied 
        by the hierarchical tree (dendrogram)
    '''
    from scipy.spatial.distance import pdist, squareform
    from fastcluster import linkage
    
    N = len(dist_mat)
    try: 
        flat_dist_mat = squareform(dist_mat)
    except Exception as e: 
        print('(compute_serial_matrix) Warning: %s' % e) 
        if isinstance(dist_mat, DataFrame):
            cols = dist_mat.columns 
            D = dist_mat.values
            np.fill_diagonal(D, 0.0)
            dist_mat = DataFrame(D, columns=cols, index=cols)
        else: 
            np.fill_diagonal(dist_mat, 0.0)
            
        flat_dist_mat = squareform(dist_mat)

    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
    
    return seriated_dist, res_order, res_linkage

####################################################################################

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])
 
def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)

def init_board(N):
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])
    return X

def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X

##### Gap Statistics 
def Wk(mu, clusters):
    K = len(mu)
    return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
               for i in range(K) for c in clusters[i]])

# limit: 2D points 
def bounding_box(X):
    xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
    ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
    return (xmin,xmax), (ymin,ymax)

def gap_statistic(X):
    (xmin,xmax), (ymin,ymax) = bounding_box(X)
    # Dispersion for real distribution
    ks = range(1,10)
    Wks = np.zeros(len(ks))
    Wkbs = np.zeros(len(ks))
    sk = np.zeros(len(ks))
    for indk, k in enumerate(ks):
        mu, clusters = find_centers(X,k)
        Wks[indk] = np.log(Wk(mu, clusters))
        # Create B reference datasets
        B = 10
        BWkbs = np.zeros(B)
        for i in range(B):
            Xb = []
            for n in range(len(X)):
                Xb.append([random.uniform(xmin,xmax),
                          random.uniform(ymin,ymax)])
            Xb = np.array(Xb)
            mu, clusters = find_centers(Xb,k)
            BWkbs[i] = np.log(Wk(mu, clusters))
        Wkbs[indk] = sum(BWkbs)/B
        sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
    sk = sk*np.sqrt(1+1/B)
    return(ks, Wks, Wkbs, sk)

def demo_spectral(): 
    import numpy as np
    from sklearn.cluster import SpectralClustering, KMeans

    mat = np.matrix([[1.,.1,.6,.4],[.1,1.,.1,.2],[.6,.1,1.,.7],[.4,.2,.7,1.]])
    SpectralClustering(2).fit_predict(mat)

    eigen_values, eigen_vectors = np.linalg.eigh(mat)
    KMeans(n_clusters=2, init='k-means++').fit_predict(eigen_vectors[:, 2:4])


    return

def test(**kargs): 

    

    return

if __name__ == "__main__":
    test() 