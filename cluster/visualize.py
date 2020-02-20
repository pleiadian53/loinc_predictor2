# encoding: utf-8

# Initialize plotting library and functions for 3D scatter plots 
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_classification, make_regression
# encoding: utf-8
import sys, os 
parentdir = os.path.dirname(os.getcwd())
sys.path.insert(0,parentdir) # include parent directory to the module search path
import utils_sys
from utils_sys import div
Domain = 'pf2'
ProjectPath = utils_sys.getProjectPath(domain=Domain, verify_=False)  # default
try: 
    ProjectPath = os.path.abspath(argv[1])   # project path: e.g. /Users/pleiades/Documents/work/data/diabetes_cf
    Domain = os.path.basename(ProjectPath)
except: 
    pass 
assert os.path.exists(ProjectPath), "Invalid project path: %s" % ProjectPath

# global 
##########################################################################################

ProjDir = ProjectPath
DataDir = os.path.join(ProjDir, 'cluster_analysis')  # values: analysis, cluster_analysis, data

##########################################################################################

from sklearn.externals import six
import pandas as pd
import numpy as np
import argparse
import json
import re
import os
import sys
import plotly
import plotly.graph_objs as go

# plotly.offline.init_notebook_mode()

"""

Reference
---------
1. Big Endian Data: Visualizing K-Means Clusters in Jupyter Notebooks 
        
        http://www.bigendiandata.com/2017-04-18-Jupyter_Customer360/

"""


def rename_columns(df, prefix='x'):
    """
    Rename the columns of a dataframe to have X in front of them

    :param df: data frame we're operating on
    :param prefix: the prefix string
    """
    df = df.copy()
    df.columns = [prefix + str(i) for i in df.columns]
    return df

def gen_data(**kargs): 
    # Create an artificial dataset with 3 clusters for 3 feature columns
    X, Y = make_classification(n_samples=100, n_classes=3, n_features=3, n_redundant=0, n_informative=3,
	                                     scale=1000, n_clusters_per_class=1)
    df = pd.DataFrame(X)
    # rename X columns
    df = rename_columns(df)
    # and add the Y
    df['y'] = Y
    print(df.head(3))

    return df

def gen_highD(**kargs):

    # create an artificial dataset with 3 clusters
    X, Y = make_classification(n_samples=100, n_classes=4, n_features=12, n_redundant=0, n_informative=12,
                                 scale=1000, n_clusters_per_class=1)
    df = pd.DataFrame(X)
    # ensure all values are positive (this is needed for our customer 360 use-case)
    df = df.abs()
    # rename X columns
    df = rename_columns(df)
    # and add the Y
    df['y'] = Y

    # split df into cluster groups
    grouped = df.groupby(['y'], sort=True)

    # compute sums for every column in every group
    sums = grouped.sum()
    print(sums)

    return 

def visualize3D_demo(df, **kargs): 
    # import plotly.graph_objs as go
    import plotly.plotly as py

    # Visualize cluster shapes in 3d.
    plotly.tools.set_credentials_file(username='tauceti53', api_key='jfLLgsRZ53u9qFf0CZ8u')

    cluster1=df.loc[df['y'] == 0]
    cluster2=df.loc[df['y'] == 1]
    cluster3=df.loc[df['y'] == 2]

    scatter1 = dict(
        mode = "markers",
        name = "Cluster 1",
        type = "scatter3d",    
        x = cluster1.as_matrix()[:,0], y = cluster1.as_matrix()[:,1], z = cluster1.as_matrix()[:,2],
        marker = dict( size=2, color='green')
    )
    scatter2 = dict(
        mode = "markers",
        name = "Cluster 2",
        type = "scatter3d",    
        x = cluster2.as_matrix()[:,0], y = cluster2.as_matrix()[:,1], z = cluster2.as_matrix()[:,2],
        marker = dict( size=2, color='blue')
    )
    scatter3 = dict(
        mode = "markers",
        name = "Cluster 3",
        type = "scatter3d",    
        x = cluster3.as_matrix()[:,0], y = cluster3.as_matrix()[:,1], z = cluster3.as_matrix()[:,2],
        marker = dict( size=2, color='red')
    )
    cluster1 = dict(
        alphahull = 5,
        name = "Cluster 1",
        opacity = .1,
        type = "mesh3d",    
        x = cluster1.as_matrix()[:,0], y = cluster1.as_matrix()[:,1], z = cluster1.as_matrix()[:,2],
        color='green', showscale = True
    )
    cluster2 = dict(
        alphahull = 5,
        name = "Cluster 2",
        opacity = .1,
        type = "mesh3d",    
        x = cluster2.as_matrix()[:,0], y = cluster2.as_matrix()[:,1], z = cluster2.as_matrix()[:,2],
        color='blue', showscale = True
    )
    cluster3 = dict(
        alphahull = 5,
        name = "Cluster 3",
        opacity = .1,
        type = "mesh3d",    
        x = cluster3.as_matrix()[:,0], y = cluster3.as_matrix()[:,1], z = cluster3.as_matrix()[:,2],
        color='red', showscale = True
    )
    layout = dict(
        title = 'Interactive Cluster Shapes in 3D',
        scene = dict(
            xaxis = dict( zeroline=True ),
            yaxis = dict( zeroline=True ),
            zaxis = dict( zeroline=True ),
        )
    )
    fig = dict( data=[scatter1, scatter2, scatter3, cluster1, cluster2, cluster3], layout=layout )
    # Use py.iplot() for IPython notebook

    fname = 'mesh3d_sample'
    fpath = os.path.join(kargs.get('path', DataDir), fname)
    plotly.offline.iplot(fig, filename=fpath)

    # (@) Send to Plotly and show in notebook
    # py.iplot(fig, filename=fpath)
    # (@) Send to broswer 
    # plot_url = py.plot(fig, filename=fname)

    return

def config(**kargs):
    if 'plot_path' in kargs: 
        path = kargs['plot_path']
        if not os.path.exists(path):
            print('(config) Creating analytics directory:\n%s\n' % path)
            os.mkdir(path)  
    return

def test(**kargs): 
    df = gen_data()
    
    path = os.path.join(os.getcwd(), 'plot'); config(plot_path=path)
    visualize3D_demo(df, path=path)

    return

if __name__ == "__main__": 
    test()


