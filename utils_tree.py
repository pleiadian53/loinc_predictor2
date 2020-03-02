# load all libraries 
import os, sys
import collections

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor    # Import decision tree classifier model
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

import numpy as np
import pandas as pd

from tabulate import tabulate
from utils_plot import saveFig
from utils_sys import sample_dict
"""


Reference
---------
    1. Decision Tree 
        https://scikit-learn.org/stable/modules/tree.html

        - Defining some of the attributes like max_depth, max_leaf_nodes, min_impurity_split, and min_samples_leaf 
          can help prevent overfitting the model to the training data.

          + min_impurity_split: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.


    2. Visualization 

        https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176

    
Resources
---------
    1. Demo of DT and visualization 

       http://benalexkeen.com/decision-tree-classifier-in-python-using-scikit-learn/

       > converting dot file to png file 

        from subprocess import call
        call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'])


"""
plotDir = os.path.join(os.getcwd(), 'plot')

def visualize(clf, feature_names, labels=['0', '1'], file_name='test', plot_dir='', ext='tif', save=True):
    from sklearn.tree import export_graphviz
    from sklearn.externals.six import StringIO  
    # from IPython.display import Image  
    import pydotplus
    
    if not plot_dir: plot_dir = os.path.join(os.getcwd(), 'plot')

    # ensure that labels are in string format 
    labels = [str(l) for l in sorted(labels)]
    
    output_path = os.path.join(plot_dir, "{}.{}".format(file_name, ext))

    # labels = ['0','1']
    # label_names = {'0': '-', '1': '+'}
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True, # node_ids=True, 
                    special_characters=True, feature_names=feature_names, class_names=labels)
    # ... class_names must be of string type

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

    if save: 
        print("(visualize) Saving decision path plot to:\n{}\n".format(output_path))
        graph.write_png(output_path)

    # Image(graph.create_png())

    return graph

def sort_path(paths, labels=[], merge_labels=False, verbose=True, verify=True):
    import operator
    
    if len(labels) == 0: 
        labels = list(paths.keys())
        if verbose: print("(sort_path) Considering labels: {}".format(labels))
            
    if not merge_labels:
        sorted_paths = {}
        for label in labels: 
            # print("... paths[label]: {}".format(paths[label]))
            sorted_paths[label] = sorted(paths[label].items(), key=operator.itemgetter(1), reverse=True)
    else: # merge and sort
        # merge path counts associated with each label => label-agnostic path counts
        paths2 = {}
        for label in labels: 
            for dseq, cnt in paths[label].items(): 
                if not dseq in paths2: paths2[dseq] = 0
                paths2[dseq] += cnt
        

        # print("... merged paths: {}".format(paths))
        sorted_paths = sorted(paths2.items(), key=operator.itemgetter(1), reverse=True)
        
        if verify:
            topk = 3 
            for i in range(topk): 
                path, cnt = sorted_paths[i][0], sorted_paths[i][1]
                
                counts = []
                for label in labels: 
                    counts.append(paths[label].get(path, 0))
                print("(sort_path) #[{}] {} | total: {} | label-dep counts: {}\n".format(i, path, cnt, counts))
                
    return sorted_paths
    
def get_lineage(tree, feature_names, mode=0, verbose=False):
    """
    Params
    ------
    mode: {'feature_only'/0, 'full'/1}
    
    
    """
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]

    # get ids of child nodes
    idx = np.argwhere(left == -1)[:,0]    
    # print("> child nodes: {}".format(idx))

    def recurse(child, left, right, lineage=None):          
        if lineage is None:
            lineage = [child]
        if child in left:  # if input child node is among the set of children_left
            parent = np.where(left == child)[0].item() # find the node ID of its parent
            split = 'l'  # left split on the parent node to get to the child
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'

        lineage.append((parent, split, threshold[parent], features[parent]))
        # path.append(features[parent])

        if parent == 0:
            lineage.reverse()  # reverse order so that the path goes from root to leaf 
            return lineage
        else:
            return recurse(parent, left, right, lineage)

    paths = {}
    if mode in ('full', 1): 
        for child in idx:
            dseq = []  # each child as its corresponding decision path
            for node in recurse(child, left, right):
                if verbose: print(node)
                
                if isinstance(node, tuple): 
                    dseq.append(node)   # 4-tuple: (parent, split, threshold[parent], features[parent])
                    
                else: # a leaf node
                    label_id = np.argmax(tree.tree_.value[node][0])
                    # print('... label: {}'.format(label_id))
                    
                    label = label_id
                    if not label in paths: paths[label] = {}
                    cnt = len(paths[label])
                    paths[label][cnt] = dseq
                
    else: # keep track of the split point only 
        for child in idx:
            dseq = [] # each child as its corresponding decision path
            for node in recurse(child, left, right):
                if verbose: print(node)
                if isinstance(node, tuple): 
                    dseq.append(node[-1]) 
                    
                else: # a leaf node
                    label_id = np.argmax(tree.tree_.value[node][0])
                    # print('... label: {}'.format(label_id))
                    
                    label, dseq = label_id, tuple(dseq)
                    if not label in paths: paths[label] = {}
                    if not dseq in paths[label]: paths[label][dseq] = 0
                    paths[label][dseq] += 1 
                    
    return paths

def count_features2(estimator, feature_names, counts={}, labels = {}, sep=' ', verbose=True):
    if len(labels) == 0: labels = {0: '-', 1: '+'}
        
    # given a tree, keep track of all its (decision) paths from root to leaves
    dpaths = get_lineage(estimator, feature_names, mode='full', verbose=False)
    
    # collect features and their thresholds
    # paths[label][cnt] = dseq
    
    for label in labels: 
        for index, dpath  in dpaths[label].items(): 
            # index: the ordinal of each decision paths with each element as a 4-tuple: (parent, split, threshold[parent], features[parent])
            assert isinstance(dpath, list), "path value is not a list? {} | value:\n{}\n".format(type(dpath), dpath)
            for entry in dpath: 
                assert isinstance(entry, tuple), "Invalid path value:\n{}\n".format(entry)
                feature, threshold = entry[-1], entry[-2]
                if not feature in counts: counts[feature] = []
                counts[feature].append(threshold)
    
    return counts

def count_features(paths, labels=[], verify=True, sep=' '):
    import collections
    
    # input 'paths' must be a label-dependent path i.e. paths as a dictionary should have labels as keys
    if len(labels) == 0: labels = list(paths.keys())
    if verify: 
        if len(labels) > 0: 
            assert set(paths.keys()) == set(labels), \
                "Inconsistent label set: {} vs {}".format(set(paths.keys()), set(labels))
    
    # merge path counts from each label => label-agnostic path counts
    paths2 = {}
    for label in labels: 
        for dseq, cnt in paths[label].items(): 
            if not dseq in paths2: paths2[dseq] = 0
            paths2[dseq] += cnt
    
    for path in paths2: 
        if isinstance(path, str): path = path.split(sep)
        # policy a. if a node appears more than once in a decision path, count only once?
        # for node in np.unique(path): 
        #     pass

        # policy b. count each occurrence
        for node, cnt in collections.Counter(path).items(): 
            if not node in counts: counts[node] = 0
            counts[node] += cnt
    return counts

def count_paths(estimator, feature_names, paths={}, counts={}, labels = {}, merge_labels=True, to_str=False, 
                sep=' ', verbose=True, index=0):  # cf: use count_paths2() to count on per-instance basis  
    """
    The decision estimator has an attribute called tree_  which stores the entire
    tree structure and allows access to low level attributes. The binary tree
    tree_ is represented as a number of parallel arrays. The i-th element of each
    array holds information about the node `i`. Node 0 is the tree's root. NOTE:
    Some of the arrays only apply to either leaves or split nodes, resp. In this
    case the values of nodes of the other type are arbitrary!

    Among those arrays, we have:
      - left_child, id of the left child of the node
      - right_child, id of the right child of the node
      - feature, feature used for splitting the node
      - threshold, threshold value at the node
      
    """
    if len(labels) == 0: labels = {0: '-', 1: '+'}
        
    # given a tree, keep track of all its (decision) paths from root to leaves
    paths_prime = get_lineage(estimator, feature_names, mode=0, verbose=verbose)
    # print("... index: {} | paths_prime: {}".format(index, paths_prime))
    if to_str:
        paths_prime2 = {}
        for label in labels: 
            paths_prime2[label] = {}
            for path, cnt in paths_prime[label].items(): 
                assert isinstance(path, tuple), "path value is not a tuple (dtype={})? {}".format(type(path), path)
                path_str = sep.join(path)
                paths_prime2[label][path_str] = cnt
        paths_prime = paths_prime2
    # print("...... index: {} | (to_str)-> paths_prime: {}".format(index, paths_prime)) # sample_dict(paths_prime[label], 5)
        
    # merge new map (of paths) with existing map (of paths)
    for label in labels:
        #assert not to_str or isinstance(next(iter(paths[label].keys())), str), \
        #    "(count_paths) Inconsistent dtype | paths[label]:\n{}\n".format(sample_dict(paths[label], 5))
        if not label in paths: paths[label] = {}
        for dseq, cnt in paths_prime[label].items():
            if to_str: assert isinstance(dseq, str)   
            if not dseq in paths[label]: paths[label][dseq] = 0
            paths[label][dseq] += cnt
                
    # print("(debug) paths[1]: {}".format(paths[1]))
    
    if verbose: 
        for label in labels: 
            print("> Label: {} | (example) decision paths:\n{}\n".format(labels[label], sample_dict(paths[label], 5)))
            
    if merge_labels: 
        # merge path counts from each label => label-agnostic path counts
        paths2 = {}
        for label in labels: 
            for dseq, cnt in paths[label].items(): 
                if not dseq in paths2: paths2[dseq] = 0
                paths2[dseq] += cnt
        paths = paths2
        
        # count feature usage: how many times was a variable used as a splitting point?
        for path in paths: 
            if isinstance(path, str): path = path.split(sep)
            assert isinstance(path, (list, tuple, np.ndarray)), "path must be in sequence dtype but got {}".format(path)
            # policy a. if a node appears more than once in a decision path, count only once?
            # for node in np.unique(path): 
            #     pass
            
            # policy b. count each occurrence
            for node, cnt in collections.Counter(path).items(): 
                if not node in counts: counts[node] = 0
                counts[node] += cnt
                
    else: 
        # count feature occurrences
        for label in labels: 
            if not label in counts: counts[label] = {} # label-dependent counts
            for path in paths[label].keys(): 
                if isinstance(path, str): path = path.split(sep)
                
                assert isinstance(path, (list, tuple, np.ndarray)), "path must be in sequence dtype but got {}".format(path)
                
                # policy b: count each occurrence
                for node, cnt in collections.Counter(path).items(): 
                    if not node in counts[label]: counts[label][node] = 0
                    counts[label][node] += cnt  
                    
    return paths, counts

def count_paths2(estimator, Xt, feature_names, labels={}, paths={}, counts={}, merge_labels=True, 
                 to_str=False, sep=' ',verbose=True):
    """
    Count decision paths with respect to input data (Xt), where the input data instances 
    are usually the test set from a train-test split: the training split is used to
    build the decision tree, whereas the test split is used to evaluate the performance 
    and count the decision paths (so that we can find out which paths are more popular than 
    the others).
    
    """
    if len(labels) == 0: labels = {0: '-', 1: '+'}
        
    lookup = {}
    features  = [feature_names[i] for i in estimator.tree_.feature]
    # print("(count_paths2) features: {}".format(features))
    
    assert isinstance(paths, dict), "Invalid dtype for decision paths: {}".format(type(paths))
    for label in labels: 
        if not label in paths: paths[label] = {}
    # if len(counts) == 0: counts = {f: 0 for f in features}
    
    if not isinstance(Xt, np.ndarray): Xt = Xt.values
    N = Xt.shape[0]
    
    node_indicator = estimator.decision_path(Xt)
    feature_index = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    # print("> n(feature_index): {}".format(len(feature_index)))

    # the leaves ids reached by each sample.
    leaf_id = estimator.apply(Xt)
    
    # print("(count_path) size(Xt): {}, dim(node_indicator): {} | n(leaf_id): {}".format(N, node_indicator.shape, len(leaf_id)))

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.
        
    # [test]
    for i in range(N): 
        #  row i, [indptr[i]:indptr[i+1]] returns the indices of elements to take from data and indices corresponding to row i.
        
        # take the i-th row 
        dseq = []
        for node_id in node_indicator.indices[node_indicator.indptr[i]:node_indicator.indptr[i+1]]: 
            dseq.append(node_id)
        # print("> sample #{} | {}".format(i, dseq)) 


    for i in range(N): 
        sample_id = i
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]
        
        dseq = [] # path to the leaf for this sample (i)
        for node_id in node_index:
            if leaf_id[sample_id] == node_id:
                label = label_id = np.argmax(estimator.tree_.value[node_id][0])

                node_descr = "label: {}".format(labels[label_id])
                lookup[node_id] = node_descr
                # print("> final NodeID[{id}: {label}]".format(id=node_id, label=labels[label_id]))
                
                dseq.append(label_id)  # labels[label_id]
                
                continue

            if (Xt[sample_id, feature_index[node_id]] <= threshold[node_id]):
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            # feature value: X[sample_id, feature[node_id]
            node_descr = "{var} {sign} {th}".format(var=features[node_id], sign=threshold_sign, th=threshold[node_id])
            lookup[node_id] = node_descr
            # counts[features[node_id]] += 1
            
            dseq.append(features[node_id])
        ### end foreach node_id ... 
        #   ... node sequence for sample (i) is determined
        
        dseq = tuple(dseq)
        # desc_seq = '> '.join([lookup[node] for node in dseq])
        
        internal_seq, label = dseq[:-1], dseq[-1]
        if not label in paths: paths[label] = {}
        if to_str: internal_seq = sep.join(internal_seq)
            
        if not internal_seq in paths[label]: paths[label][internal_seq] = 0
        paths[label][internal_seq] += 1
        
    ### end foreach test instance
        
    if merge_labels: 
        # merge path counts from each label => label-agnostic path counts
        paths2 = {}
        for label in labels: 
            for dseq, cnt in paths[label].items(): 
                if not dseq in paths2: paths2[dseq] = 0
                paths2[dseq] += cnt
        paths = paths2
        
        # count feature usage: how many times was a variable used as a splitting point?
        for path in paths: 
            if isinstance(path, str): path = path.split(sep)
            assert isinstance(path, (list, tuple, np.ndarray)), "path must be in sequence dtype but got {}".format(path)
            # policy a. if a node appears more than once in a decision path, count only once?
            # for node in np.unique(path): 
            #     pass
            
            # policy b. count each occurrence
            for node, cnt in collections.Counter(path).items(): 
                if not node in counts: counts[node] = 0
                counts[node] += cnt
    else: 
        # count feature occurrences
        for label in labels: 
            if not label in counts: counts[label] = {} # label-dependent counts
            for path in paths[label].keys(): 
                if isinstance(path, str): path = path.split(sep)
                
                assert isinstance(path, (list, tuple, np.ndarray)), "path must be in sequence dtype but got {}".format(path)
                
                # policy b: count each occurrence
                for node, cnt in collections.Counter(path).items(): 
                    if not node in counts[label]: counts[label][node] = 0
                    counts[label][node] += cnt  
                    
    return paths, counts


def t_vis_tree(dtree):
    from sklearn.externals.six import StringIO  
    from IPython.display import Image  
    from sklearn.tree import export_graphviz
    import pydotplus   # use pip install pydotplus

    iris=datasets.load_iris()
    df=pd.DataFrame(iris.data, columns=iris.feature_names)
    y=iris.target

    dot_data = StringIO()
    export_graphviz(dtree, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    Image(graph.create_png())

    return


def t_classification(): 
    from sklearn.tree import DecisionTreeClassifier    # Import decision tree classifier model
    

    tree = DecisionTreeClassifier(criterion='entropy', # Initialize and fit classifier
        max_depth=4, random_state=1)
    tree.fit(X, y)

    t_vis_tree(tree)

    return

def test(**kargs): 

    load_merge(vars_matrix='exposures-4yrs.csv', label_matrix='nasal_biomarker_asthma1019.csv')

    return

if __name__ == "__main__": 
    test()

