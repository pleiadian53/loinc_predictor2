# import plotly.plotly as py
# import plotly.graph_objs as go

# import matplotlib.pyplot as plt 

# non-interactive mode
import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt

import pandas as pd
from pandas import DataFrame, Series
# import cufflinks as cf  # cufflinks binds plotly to pandas
import numpy as np

# import heatmap
# from sns import heatmap

import random
import os

#######################################################################
#
#
#  References
#  ----------
#  1. cufflinks: https://github.com/santosjorge/cufflinks
#
#
#

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_path(name='test', basedir=None, ext='tif', create_dir=False):
    import os
    # create the desired path to the plot by its name
    if basedir is None: basedir = os.path.join(os.getcwd(), 'plot')
    if not os.path.exists(basedir) and create_dir:
        print('(plot) Creating plot directory:\n%s\n' % basedir)
        os.mkdir(basedir) 
    return os.path.join(basedir, '%s.%s' % (name, ext))

def saveFig(plt, fpath, ext='tif', dpi=500, message='', verbose=True):
    """
    fpath: 
       name of output file
       full path to the output file

    Memo
    ----
    1. supported graphing format: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff 

    """
    import os
    outputdir, fname = os.path.dirname(fpath), os.path.basename(fpath) 

    # [todo] abstraction
    supported_formats = ['eps', 'jpeg', 'jpg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff', ]  

    # supported graphing format: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    if not outputdir: outputdir = os.getcwd() # sys_config.read('DataExpRoot') # ./bulk_training/data-learner)
    assert os.path.exists(outputdir), "Invalid output path: %s" % outputdir
    ext_plot = ext  # eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    if not fname: fname = 'generic-test.%s' % ext_plot
    fbase, fext = os.path.splitext(fname)
    assert fext[1:] in supported_formats, "Unsupported graphic format: %s" % fname

    fpath = os.path.join(outputdir, fname)

    if verbose: print('(saveFig) Saving plot to:\n%s\n... description: %s' % (fpath, 'n/a' if not message else message))
    
    # pylab leaves a generous, often undesirable, whitespace around the image. Remove it by setting bbox_inches to tight
    plt.savefig(fpath, bbox_inches='tight', dpi=dpi)   
    return

def t_heatmap(): 
    import heatmap
    import random

    pts = []
    for x in range(400):
        pts.append((random.random(), random.random() ))

    hm = heatmap.Heatmap()
    img = hm.heatmap(pts)
    img.save("classic.png")

    return

def t_histogram(**kargs): 

    ### plot multiple histograms

    # t_plot_multiple_hist(**kargs)

    ### 
    # t_plot_dataframe(**kargs)

    ### Bar plot 
    y = [0.80, 0.80, 0.89, 0.78, 0.73, 0.71, 0.79, 0.82, 0.98, 0.86, 0.82]
    x = ['G1-control', 'G1A1-control',  'Stage 1', 'Stage 2', 'Stage 3a', 'Stage 3b', 
         'Stage 4', 'Stage 5', 'ESRD after transplant', 'ESRD on dialysis', 'Unknown']

    # a wrapper for t_plotly()
    makeBarPlot(x, y, title_x="CKD stages", title_y="Area under the curve (AUC)", color_marker='#C8E3EF', cohort='CKD')

    return

def t_colors(**kargs):
    from itertools import cycle, islice
    import pandas, numpy as np  # I find np.random.randint to be better

    # 
    factor = 'color'
    basedir = os.getcwd() # or sys_config.read('DataExpRoot')  # 'seqmaker/data/CKD/plot'

    # Make the data
    x = [{i:np.random.randint(1,5)} for i in range(10)]
    df = pandas.DataFrame(x)

    # Make a list by cycling through the colors you care about
    # to match the length of your data.
    my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(df)))

    # Specify this list of colors as the `color` option to `plot`.
    df.plot(kind='bar', stacked=True, color=my_colors) 
    plt.savefig(os.path.join(basedir, 'test-%s.tif' % factor ))  #

    return

