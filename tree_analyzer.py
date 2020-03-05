import os, sys
import collections, random

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor    # Import decision tree classifier model
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

import numpy as np
import pandas as pd

from tabulate import tabulate
from utils_plot import saveFig
from analyzer import run_model_selection
import common
"""


Reference
---------
    1. Decision Tree 
        https://scikit-learn.org/stable/modules/tree.html

    2. Visualization 

        https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176

    3. Nested vs non-nested: 

        https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#id2

"""
dataDir = os.path.join(os.getcwd(), 'data')  # default unless specified otherwise
plotDir = os.path.join(os.getcwd(), 'plot')

class Data(object): 
    label = 'label'
    features = []

# convenient wrapper for DT classifier 
def classify(X, y, params={}, random_state=0, **kargs): 
    assert isinstance(params, dict)
    info_gain_measure = kargs.get('criterion', 'entropy')
    model = DecisionTreeClassifier(criterion=info_gain_measure, random_state=random_state)
    if len(params) > 0: model = model.set_params(**params)
    model.fit(X, y)
    return model

# convenient wrapper for DT classifier 
def apply_model(X, y, model=None, params={}, **kargs): 
    assert isinstance(params, dict)
    if model is None: 
        info_gain_measure = kargs.get('criterion', 'entropy')
        random_state = kargs.get('random_state', 53)
        model = DecisionTreeClassifier(criterion=info_gain_measure, random_state=random_state)
    if len(params) > 0: model = model.set_params(**params)
    model.fit(X, y)
    return model

def predict(X, y=[], p_th=0.5, model=None, params={}, **kargs): 
    assert isinstance(params, dict)
    if model is None: 
        info_gain_measure = kargs.get('criterion', 'entropy')
        random_state = kargs.get('random_state', 53)
        model = DecisionTreeClassifier(criterion=info_gain_measure, random_state=random_state)
    y_pred = model.predict_proba(X)
    return y_pred

def analyze_path(X, y, model=None, p_grid={}, best_params={}, feature_set=[], n_trials=100, n_trials_ms=10, save=True, output_path='', output_file='', 
                    create_dir=True, index=0, **kargs):
    # from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
    from sklearn.model_selection import train_test_split # Import train_test_split function
    from utils_tree import visualize, count_paths, count_paths2, count_features2
    import time
    from analyzer import det_range, det_range2
    
    #### parameters ####
    test_size = kargs.get('test_size', 0.2)
    verbose = kargs.get('verbose', False)
    merge_labels = kargs.get('merge_labels', True)
    policy_count = kargs.get('policy_counts', 'standard') # options: {'standard', 'sample-based'}
    to_str = kargs.get('to_str', False)  # if True, decision paths are represented by strings (instead of tuples)
    
    # [output] tree visualization
    experiment_id = kargs.get('experiment_id', 'test') # a file ID for the output (e.g. example decision tree)
    validate_tree = kargs.get('save_tree', True)

    # [output] model selection
    plot_dir = kargs.get('plot_dir', plotDir)
    plot_ext = kargs.get('plot_ext', 'pdf')
    validate_ms = kargs.get('save_ms', True)

    # output meta generated from model training
    tReturnMetaData = kargs.get('return_meta_data', False)

    ####################
    
    labels = np.unique(y)
    N, Nd = X.shape
    
    if len(feature_set) == 0: feature_set = ['f%s' % i for i in range(Nd)]
        
    msg = ''
    if verbose: 
        msg += "(analyze_path) dim(X): {} | vars (n={}):\n...{}\n".format(X.shape, len(feature_set), feature_set)
        msg += "... class distribution: {}\n".format(collections.Counter(y))
    print(msg)

    # -- define model 
    if model is None: 
        info_gain_measure = kargs.get('criterion', 'entropy')
        model = DecisionTreeClassifier(criterion=info_gain_measure, random_state=time.time())
    
    # -- run model selection
    if len(p_grid) > 0: 
        best_params, params, nested_scores = \
            run_model_selection(X, y, model, p_grid=p_grid, n_trials=n_trials_ms, 
                output_path=output_path, meta=experiment_id, ext=plot_ext, save=validate_ms)
        # best_index = np.argmax(nested_scores)
        # best_params = params[best_index]
        print('[path] best_params(DT):\n{}\n'.format(best_params))
    else: 
        assert len(best_params) > 0
        
    # initiate data structures 
    paths = {}
    lookup = {}
    counts = {f: [] for f in feature_set} # maps features to lists of thresholds
        
    # build N different decision trees and compute their statistics (e.g. performance measures, decision path counts)
    measures = ['accuracy', 'auc', 'fmax', 'p_threshold']
    scores = {m:[] for m in measures}
    test_points = np.random.choice(range(n_trials), 1)

    # prob_thresholds = []
    for i in range(n_trials): 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i) # 70% training and 30% test
        print("[{}] dim(X_test): {}".format(i, X_test.shape))

        # [test] feature value range
        # print("[test] Feature value range ...")
        # det_range2(X_train, y_train, fset=feature_set, pos_label=1)

        # model = model.set_params(**best_params).fit(X_train, y_train)
        model = apply_model(X_train, y_train, model=model, params=best_params) 
        # ... fit (X, y) at given "params" (e.g. best params from model selection)
        
        # [test]
        if i in test_points: 
            if verbose: print("... building {} versions of the model: {}".format(n_trials, model.get_params()) )
            if validate_tree: 
                fname = "{id}-{index}".format(id=experiment_id, index=i)
                graph = visualize(model, feature_set, labels, file_name=fname, ext=plot_ext, save=validate_tree)
                
                # display the tree in the notebook
                # Image(graph.create_png())  # from IPython.display import Image
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)

        # need probabilistic predictions for the following metrics
        auc_score = metrics.roc_auc_score(y_test, y_prob[:,1])
        fmax, p_th = common.fmax_score_threshold(y_test, y_prob[:,1])  
        # ... Note that the index of the class: clf.classes_

        print("[{}] F1 reaches max: {} at p_th: {}".format(i, fmax, p_th))
        if i % 10 == 0: 
            print("[{}] Accuracy: {}, AUC: {}, Fmax: {}".format(i, accuracy, auc_score, fmax))
        scores['accuracy'].append(accuracy); scores['auc'].append(auc_score); scores['fmax'].append(fmax)
        scores['p_threshold'].append(p_th)

        if not isinstance(X_test, np.ndarray): X_test = X_test.values   
            
        # --- count paths ---
        #    method A: count number of occurrences of decision paths read off of the decision tree
        #    method B: sample-based path counts
        #              each X_test[i] has its associated decision path => 
        #              in this method, we count the number of decision paths wrt the test examples
        if policy_count.startswith('stand'): # 'standard'
            paths, _ = \
                count_paths(model, feature_names=feature_set, paths=paths, # count_paths, 
                            merge_labels=merge_labels, to_str=to_str, verbose=False, index=i)
                # ... set verbose to True to print lineage 
        else:  # 'full'
            paths, _ = \
                count_paths2(model, X_test, feature_names=feature_set, paths=paths, # counts=counts, 
                             merge_labels=merge_labels, to_str=to_str, verbose=False)
            
        # keep track of feature usage in terms of thresholds at splitting points
        counts = count_features2(model, feature_names=feature_set, counts=counts, labels=labels, verbose=True)
        # ... counts: feature -> list of thresholds (used to estimate its median across decision paths)

        # visualization?
    ### end foreach trial 
    print("\n(analyze_path) E[Acc]: {} | E[AUC]: {} | E[Fmax]: {} | n_trials={}".format(np.mean(scores['accuracy']),
        np.mean(scores['auc']), np.mean(scores['fmax']), n_trials))
    
    if tReturnMetaData: 
        return paths, counts, scores         

    return paths, counts

def load_data(cohort, **kargs): 
    """

    Memo
    ----
    1. Example datasets

        a. multivariate imputation applied 
            exposures-4yrs-merged-imputed.csv
        b. rows with 'nan' dropped 
            exposures-4yrs-merged.csv

    """
    import data_processor as dp
    return dp.load_data(cohort, **kargs)

def runWorkflow(X, y, features, model=None, param_grid={}, **kargs):
    # 2. define model (e.g. decision tree)
    if verbose: print("(runWorkflow) 2. Define model (e.g. decision tree and its parameters) ...")
    ######################################################
    
    ######################################################
    # ... min_samples_leaf: the minimum number of samples required to be at a leaf node. 
    #     A split point at any depth will only be considered if it leaves at least 
    #     min_samples_leaf training samples in each of the left and right branches
    counts = {}
    experiment_id = kargs.get('experiment_id', 'test')

    if not model: 
        if not param_grid:  # p_grid
            param_grid = {"max_depth": [3, 4, 5, 8, 10, 15], 
                          "min_samples_leaf": [1, 5, 10, 15, 20]}

        model = DecisionTreeClassifier(criterion='entropy', random_state=1)

        ###################################################### 
        test_size = 0.3
        rs = 53
        topk = 10
        topk_vars = 10
        ######################################################
        labels = [str(l) for l in sorted(np.unique(y))]

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs) # 70% training and 30% test

        # params = {'max_depth': 5}  # use model selection to determine the optimal 'max_depth'
        # model = classify(X_train, y_train, params=params, random_state=rs)
        # graph = visualize(model, features, labels=labels, plot_dir=plotDir, file_name=file_prefix, ext='tif')

        # 4. analyze decision paths and keep track of frequent features
        paths, counts = \
            analyze_path(X, y, model=model, p_grid=param_grid, feature_set=features, n_trials=100, n_trials_ms=30, save=False,  
                            merge_labels=False, policy_count='standard', experiment_id=experiment_id,
                               create_dir=True, index=0, validate_tree=False, to_str=True, verbose=False)

        summarize_paths(paths, topk=topk)

        # for k, ths in counts.items(): 
        #     assert isinstance(ths, list), "{} -> {}".format(k, ths)
    if counts: 
        msg = "(runWorkflow) Count feature usage (applicable for tree-based methods e.g. decision tree) ...\n"
        fcounts = [(k, len(ths)) for k, ths in counts.items()]
        sorted_features = sorted(fcounts, key=lambda x: x[1], reverse=True)
        msg += "> Top {} features:\n{}\n".format(topk_vars, sorted_features[:topk_vars])
        print(msg)

    return model

if __name__ == "__main__":
    runWorkflow()