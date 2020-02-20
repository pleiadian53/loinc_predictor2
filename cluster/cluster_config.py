# encoding: utf-8
import sys, os 
parentdir = os.path.dirname(os.getcwd())
sys.path.insert(0,parentdir) # include parent directory to the module search path
import utils_sys
from utils_sys import div
Domain = 'loinc'

import pandas as pd
from pandas import DataFrame, Series

# import matplotlib
# matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
# from matplotlib import pyplot as plt

import os, sys, collections, re, glob
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

class TSet(object):
    index_field = 'person_id'  
    date_field = 'date'
    target_field = 'target'  # usually surrogate labels
    annotated_field = 'annotated'
    content_field = 'content'  # representative sequence elements 
    label_field = 'mlabels'  # multiple label repr of the underlying sequence (e.g. most frequent codes as constituent labels)

    meta_fields = [target_field, index_field, ]

def name_image_file(descriptor, **kargs):  # e.g. heatmap
    ext = kargs.get('graph_ext', 'tif')  

    n_sample = kargs.get('n_sample', kargs.get('n_points', None))
    seq_ptype = kargs.get('seq_ptype', 'regular')
    d2v_method = kargs.get('d2v_method', 'PVDM')
    # vis_method = kargs.get('descriptor', 'manifold') # visualization method
    identifier = '%s_%s_%s' % (descriptor, seq_ptype, d2v_method)

    if n_sample is not None: 
        return 'I%s-S%d.%s' % (identifier, n_sample, ext)
    return 'I%s.%s' % (identifier, ext)

def name_cluster_file(descriptor, **kargs):
    ext = kargs.get('graph_ext', 'tif')

    n_sample = kargs.get('n_sample', kargs.get('n_points', None))
    seq_ptype = kargs.get('seq_ptype', 'regular')
    d2v_method = kargs.get('d2v_method', 'PVDM')
    # vis_method = kargs.get('descriptor', 'manifold') # visualization method
    identifier = '%s_%s_%s' % (descriptor, seq_ptype, d2v_method)

    if n_sample is not None: 
        return '%s-S%d.%s' % (identifier, n_sample, ext)
    return '%s.%s' % (identifier, ext)

def test(**kargs): 
    setting = "(cluster_config) module: {module} | domain: {dataset}, project_path: {path}".format(module=__name__, dataset=Domain, path=ProjectPath)
    div(setting, symbol='=', border=2)  
  
    return


if __name__ == "__main__":
    test()
