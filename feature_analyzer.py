# encoding: utf-8

import os
from pandas import DataFrame
import pandas as pd

import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
import matplotlib.pyplot as plt

# select plotting style; must be called prior to pyplot
plt.style.use('seaborn')  # values: {'seaborn', 'ggplot', }  
from utils_plot import saveFig

import seaborn as sns
import numpy as np 

from tabulate import tabulate

"""

Reference
---------
1. Better heatmap and correlation matrix plot: 

        https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
"""

def plot_data_matrix(df, output_path='', dpi=300, **kargs):
    plt.clf()

    print('(plot_similarity_matrix) dim(df): {d}, columns (n={n}): {cols}'.format(d=df.shape, n=len(df.columns.values), cols=df.columns.values))

    verbose = kargs.get('verbose', 1)
    dpi = kargs.get('dpi', 300)

    # mask upper triangle? 
    tMaskUpper = kargs.get('mask_upper', False)

    # range
    vmin, vmax = kargs.get('vmin', 0.0), kargs.get('vmax', 1.0)  # similarity >= 0
    annotation = kargs.get('annot', False)

    # Generate a mask for the upper triangle
    mask = None
    if tMaskUpper: 
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

    df = pd.melt(df.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
          
    # e.g.  
    #        index                   variable     value
    # 0          0  test_order_name_SHORTNAME  0.000000
    # 1          1  test_order_name_SHORTNAME  0.047290
    # 2          2  test_order_name_SHORTNAME  0.000000
    # 3          3  test_order_name_SHORTNAME  0.092474
    # 4          4  test_order_name_SHORTNAME  0.000000
    # ...      ...                        ...       ...

    df.columns = ['y', 'x', 'value']
    # set variabes in the x axis
    #     index    in the y axis

    ax = heatmap(
            x=df['x'],
            y=df['y'],
            size=df['value'].abs(),
            color=df['value'], 
            
            # vmin=vmin, vmax=vmax, center=0,
            # annot=annotation,

            # --- other options 
            # color_range=[-1, 1],
            # palette
            # size 
            # size_range
    )

    if not output_path: output_path = os.path.join('plot', "data_matrix.pdf")
    assert os.path.exists(os.path.dirname(output_path)), "Invalid path: {}".format(os.path.dirname(output_path))
    if verbose: print('(plot_data_matrix) Saving heatmap at: {path}'.format(path=output_path))
    saveFig(plt, output_path, dpi=dpi)

    return 

def plot_similarity_matrix(S=None, data=None, output_path='', **kargs): 
    plt.clf()

    if data is not None: 
        S = data.corr() # original data, not a similarity matrix
        # e.g. pd.read_csv('https://raw.githubusercontent.com/drazenz/heatmap/master/autos.clean.csv')
        
    elif S is not None: 
        assert isinstance(S, DataFrame)
    print('(plot_similarity_matrix) columns (n={n}): {cols}'.format(n=len(data.columns.values), cols=data.columns.values))

    verbose = kargs.get('verbose', 1)
    dpi = kargs.get('dpi', 300)
      
    # mask upper triangle? 
    tMaskUpper = kargs.get('mask_upper', False)

    # range
    vmin, vmax = kargs.get('vmin', 0.0), kargs.get('vmax', 1.0)  # similarity >= 0
    annotation = kargs.get('annot', False)

    # Generate a mask for the upper triangle
    mask = None
    if tMaskUpper: 
        mask = np.zeros_like(S, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # ax = sns.heatmap(
    #     S, 
    #     vmin=vmin, vmax=vmax, center=0,
    #     cmap=sns.diverging_palette(20, 220, n=200),  # sns.diverging_palette(220, 10, as_cmap=True)
    #     square=True, 
    #     annot=annotation,
    #     mask=mask,  
    # )
    # ax.set_xticklabels(
    #     ax.get_xticklabels(),
    #     rotation=45,
    #     horizontalalignment='right'
    # );

    # use enhanced heatmap()

    # filter columns 

    S = pd.melt(S.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
    S.columns = ['x', 'y', 'value']

    ax = heatmap(
            x=S['x'],
            y=S['y'],
            size=S['value'].abs(),
            color=S['value'], 
            
            # vmin=vmin, vmax=vmax, center=0,
            # annot=annotation,

            # --- other options 
            # color_range=[-1, 1],
            # palette
            # size 
            # size_range
    )

    if not output_path: output_path = os.path.join('plot', "sim_matrix.pdf")
    assert os.path.exists(os.path.dirname(output_path)), "Invalid path: {}".format(os.path.dirname(output_path))
    if verbose: print('(plot_similarity_matrix) Saving heatmap at: {path}'.format(path=output_path))
    saveFig(plt, output_path, dpi=dpi)

    return 

# --- alias
plot_heatmap = plot_similarity_matrix

def heatmap(x, y, **kwargs):
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)
        
    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        # palette = sns.color_palette("Blues", n_colors) 
        palette = sns.diverging_palette(20, 220, n=n_colors) # Create the palette
        
    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]
    
    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)
    
    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)
        
    size_scale = kwargs.get('size_scale', 500)
    
    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale
    
    if 'x_order' in kwargs: 
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}
    
    if 'y_order' in kwargs: 
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}
    
    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot
        
    marker = kwargs.get('marker', 's')
    
    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order'
    ]}
    
    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size], 
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])
    
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')
    
    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot

        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

        bar_height = y[1] - y[0]
        ax.barh(
            y=y,
            width=[5]*len(palette), # Make bars 5 units wide
            left=col_x, # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False) # Hide grid
        ax.set_facecolor('white') # Make background white
        ax.set_xticks([]) # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(y), max(y), 3)) # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right() # Show vertical ticks on the right 
    return ax
    
def corrplot(data, size_scale=500, marker='s'):
    corr = pd.melt(data.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    heatmap(
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr['value'].abs(), size_range=[0,1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale
    )

def runClustering(A, n_clusters=-1, method='kmeans', **kargs): 
    import cluster.cluster as cluster

    if method == 'kmeans': 
        return kmeansCluster(A, n_clusters, **kargs)
    elif method.startswith('spe'): 
        return spectralCluster(A, n_clusters, **kargs)

    # else try to find the function in a particular module 

    tMethodSupported = False
    naming_protocal = '{base}Cluster'.format(base=method)
    for clustering_method in [method, naming_protocal, ]:
        try: 
            clustering_func = getattr(cluster, clustering_method)
            if hasattr(clustering_func, '__call__'): 
                tMethodSupported = True
        except: 
             pass
    if tMethodSupported: 
        return clustering_func(A, n_clusters=n_clusters, **kargs)

    # last resort is to find a supporting function in the current globals namespace
    return globals()[naming_protocal](A, n_clusters=n_clusters, **kargs)

def spectralCluster(A, n_clusters=-1, **kargs):  # A: latent factors P, Q
    import cluster.cluster as cluster
    # from scipy.spatial import distance   # todo: use angular distance
    if n_clusters == -1: 
        n_clusters = A.shape[1]  # dimension of the latent factor matrix (e.g. user matrix, item matrix)

    if kargs.get('verbose', True): print('(clustering) method: spectral, n_clusters: {0} | dim(A): {1}'.format(n_clusters, A.shape))
    S = toAffinity(A, sig=kargs.get('bandwith', 0.5)) # evalSimilarityByLatentFeatures(D) 
    return cluster.spectralCluster(X=S, n_clusters=n_clusters)  # return cluster label IDs (a numpy.ndarray)

def kmeansCluster(X, n_clusters=-1, **kargs): 
    import cluster.cluster as cluster
    # from scipy.spatial import distance   # todo: use angular distance
    if n_clusters == -1: 
        n_clusters = X.shape[1]  # dimension of the latent factor matrix (e.g. user matrix, item matrix)

    if kargs.get('verbose', True): print('(clustering) method: kmeans, n_clusters: {0} | dim(A): {1}'.format(n_clusters, X.shape))
    return cluster.kmeansCluster(X, n_clusters=n_clusters)

def toAffinity(A, sim_func=None, sig=0.5, verify=False):
    if sim_func is None: sim_func = evalPairwiseSimilarity  # similarity falls in [0, 1]

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
def evalPairwiseSimilarity(A, epsilon=1e-9):
    """
    Compute pairwise similarity between "rows" of A (i.e. assuming A is in row vector format). 

    """
    from sklearn.preprocessing import normalize

    A = normalize(A, axis=1, norm='l2')

    # Below is NOT recommmended see Memo [1]
    # sim = np.dot(A, A.T) # A.dot(A.T) # + epsilon
    # norms = np.array([np.sqrt(np.diagonal(sim))])
    # return (sim / norms / norms.T) 

    return np.dot(A, A.T)


def t_cluster(): 
    from sklearn.datasets.samples_generator import make_blobs

    n_users = 10
    nf = 5
    X, y = make_blobs(n_samples=n_users, centers=3, n_features=nf, random_state=0)

    U = ['user%d' % (i+1) for i in range(n_users)]

    print('(t_cluster) running cluster analysis ...')
    labels =  runClustering(X, n_clusters=3, method='spectral')
    S = evalPairwiseSimilarity(X, epsilon=1e-9) 

    n_display = 10
    df = DataFrame(S, columns=U, index=U)
    tabulate(df.head(n_display), headers='keys', tablefmt='psql')

    print('(t_cluster) Plotting similarity matrix ...')
    parentdir = os.path.dirname(os.getcwd())
    testdir = os.path.join(parentdir, 'test')  # e.g. /Users/<user>/work/data

    fpath = os.path.join(testdir, 'heatmap-cluster-sim.png') 
    # ... tif may not be supported (Format 'tif' is not supported (supported formats: eps, pdf, pgf, png, ps, raw, rgba, svg, svgz))

    plot_heatmap(data=df, output_path=fpath)

    return
    
def t_heatmap(corr=None, **kargs):
    import seaborn as sns 

    plt.clf()
    mode = 'conventional' # 'conventional', 'enchanced'

    # corr: similarity_dataframe
    if corr is None: 
        data = pd.read_csv('https://raw.githubusercontent.com/drazenz/heatmap/master/autos.clean.csv')
        corr = data.corr()

    if mode.startswith('conv'): 
        

        # --- Conventional heatmap
        ax = sns.heatmap(
            corr, 
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        );
    else: 
        corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
        corr.columns = ['x', 'y', 'value']
        heatmap(
            x=corr['x'],
            y=corr['y'],
            size=corr['value'].abs()
        ) 

    parentdir = os.path.dirname(os.getcwd())
    testdir = os.path.join(parentdir, 'test')  # e.g. /Users/<user>/work/data
    output_path = os.path.join(testdir, 'heatmap-test.png') 
    saveFig(plt, output_path, dpi=200)

    return


def test(): 
    import pandas as pd 

    parentdir = os.path.dirname(os.getcwd())
    testdir = os.path.join(parentdir, 'test')  # e.g. /Users/<user>/work/data

    ### plot heatmap
    # df = pd.read_csv('https://raw.githubusercontent.com/drazenz/heatmap/master/autos.clean.csv')
    # fpath = os.path.join(testdir, 'heatmap-1.tif')
    # plot_heatmap(data=df, output_path=fpath)
    # t_heatmap()

    t_cluster()

    return


if __name__ == "__main__":
    test()