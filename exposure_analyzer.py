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


def run_model_selection(X, y, model, p_grid={}, n_trials=30, scoring='roc_auc', output_path='', output_file='', create_dir=True, 
                        index=0, plot_=True, ext='tif', save=False): 
    from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
    
    # Arrays to store scores
    non_nested_scores = np.zeros(n_trials)
    nested_scores = np.zeros(n_trials)

    # Loop for each trial
    icv_num = 5
    ocv_num = 5
    best_params = {}
    for i in range(n_trials):

        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset.
        # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
        inner_cv = KFold(n_splits=icv_num, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=ocv_num, shuffle=True, random_state=i)

        # Non_nested parameter search and scoring
        clf = GridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv,
                           iid=False)

        clf.fit(X, y)
        non_nested_scores[i] = clf.best_score_

        # Nested CV with parameter optimization
        nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, scoring=scoring)
        nested_scores[i] = nested_score.mean()
        best_params[i] = clf.best_params_

    score_difference = non_nested_scores - nested_scores

    print("Average difference of {:6f} with std. dev. of {:6f}."
          .format(score_difference.mean(), score_difference.std()))

    if plot_: 
        plt.clf()
        
        # Plot scores on each trial for nested and non-nested CV
        plt.figure()
        plt.subplot(211)
        non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
        nested_line, = plt.plot(nested_scores, color='b')
        plt.ylabel("score", fontsize="14")
        plt.legend([non_nested_scores_line, nested_line],
                   ["Non-Nested CV", "Nested CV"],
                   bbox_to_anchor=(0, .4, .5, 0))
        plt.title("Non-Nested and Nested Cross Validation",
                  x=.5, y=1.1, fontsize="15")

        # Plot bar chart of the difference.
        plt.subplot(212)
        difference_plot = plt.bar(range(n_trials), score_difference)
        plt.xlabel("Individual Trial #")
        plt.legend([difference_plot],
                   ["Non-Nested CV - Nested CV Score"],
                   bbox_to_anchor=(0, 1, .8, 0))
        plt.ylabel("score difference", fontsize="14")

        if save: 
            from utils_plot import saveFig
            if not output_path: output_path = os.path.join(os.getcwd(), 'analysis')
            if not os.path.exists(output_path) and create_dir:
                print('(run_model_selection) Creating analysis directory:\n%s\n' % output_path)
                os.mkdir(output_path) 

            if output_file is None: 
                classifier = 'DT'
                name = 'ModelSelect-{}'.format(classifier)
                suffix = n_trials 
                output_file = '{prefix}.P-{suffix}-{index}.{ext}'.format(prefix=name, suffix=suffix, index=index, ext=ext)

            output_path = os.path.join(output_path, output_file)  # example path: System.analysisPath

            if verbose: print('(run_model_selection) Saving model-selection-comparison plot at: {path}'.format(path=output_path))
            saveFig(plt, output_path, dpi=dpi, verbose=verbose) 
        else: 
            plt.show()
        
    return best_params, nested_scores

# convenient wrapper for DT classifier 
def classify(X, y, params={}, random_state=0, **kargs): 
    assert isinstance(params, dict)
    info_gain_measure = kargs.get('criterion', 'entropy')
    model = DecisionTreeClassifier(criterion=info_gain_measure, random_state=random_state)
    if len(params) > 0: model = model.set_params(**params)
    model.fit(X, y)

    return model

def analyze_path(X, y, model=None, p_grid={}, feature_set=[], n_trials=100, n_trials_ms=10, save=False, output_path='', output_file='', 
                             create_dir=True, index=0, **kargs):
    # from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
    from sklearn.model_selection import train_test_split # Import train_test_split function
    from utils_tree import visualize, count_paths, count_paths2, count_features2
    import time
    
    #### parameters ####
    test_size = kargs.get('test_size', 0.2)
    verbose = kargs.get('verbose', False)
    merge_labels = kargs.get('merge_labels', True)
    policy_count = kargs.get('policy_counts', 'standard') # options: {'standard', 'sample-based'}
    experiment_id = kargs.get('experiment_id', 'test') # a file ID for the output (e.g. example decision tree)
    validate_tree = kargs.get('validate_tree', True)
    plot_dir = kargs.get('plot_dir', plotDir)
    plot_ext = kargs.get('plot_ext', 'tif')
    to_str = kargs.get('to_str', False)  # if True, decision paths are represented by strings (instead of tuples)
    ####################
    
    labels = np.unique(y)
    N, Nd = X.shape
    
    if len(feature_set) == 0: feature_set = ['f%s' % i for i in range(Nd)]
        
    msg = ''
    if verbose: 
        msg += "(analyze_path) dim(X): {} | vars (n={}):\n...{}\n".format(X.shape, len(feature_set), feature_set)
        msg += "... class distribution: {}\n".format(collections.Counter(y))
    print(msg)

    # define model 
    if model is None: model = DecisionTreeClassifier(criterion='entropy', random_state=time.time())
    
    # run model selection 
    if len(p_grid) > 0: 
        best_params, nested_scores = \
          run_model_selection(X, y, model, p_grid=p_grid, n_trials=n_trials_ms, output_path=output_path, ext=plot_ext)
        the_index = np.argmax(nested_scores)
        the_params = best_params[np.argmax(nested_scores)]
        # print('> type(best_params): {}:\n{}\n'.format(type(best_params), best_params))
        
    # initiate data structures 
    paths = {}
    lookup = {}
    counts = {f: [] for f in feature_set} # maps features to lists of thresholds
        
    # build N different decision trees and compute their statistics (e.g. performance measures, decision path counts)
    auc_scores = []
    test_points = np.random.choice(range(n_trials), 1)
    for i in range(n_trials): 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i) # 70% training and 30% test
        # print("[{}] dim(X_test): {}".format(i, X_test.shape))

        # [todo]: how to reset a (trained) model? 
        model = classify(X_train, y_train, params=the_params, random_state=i)
        
        # [test]
        if i in test_points: 
            if verbose: print("... building {} versions of the model: {}".format(n_trials, model.get_params()) )
            if validate_tree: 
                fild_prefix = "{id}-{index}".format(id=experiment_id, index=i)
                graph = visualize(model, feature_set, labels, file_name=file_prefix, ext='tif')
                
                # display the tree in the notebook
                # Image(graph.create_png())  # from IPython.display import Image
        
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        auc_score = metrics.roc_auc_score(y_test, y_pred)
        if i % 10 == 0: print("[{}] Accuracy: {}, AUC: {}".format(i, accuracy, auc_score))
        auc_scores.append(auc_score)

        if not isinstance(X_test, np.ndarray): X_test = X_test.values   
            
        # --- count paths ---
        #    method A: count number of occurrences of decision paths read off of the decision tree
        #    method B: sample-based path counts
        #              each X_test[i] has its associated decision path => 
        #              in this method, we count the number of decision paths wrt the test examples
        if policy_count.startswith('stand'): # 'standard'
            paths, _ = \
                count_paths(model, feature_names=feature_set, paths=paths, # count_paths, 
                            merge_labels=merge_labels, to_str=to_str, verbose=verbose, index=i)
        else:  # 'full'
            paths, _ = \
                count_paths2(model, X_test, feature_names=feature_set, paths=paths, # counts=counts, 
                             merge_labels=merge_labels, to_str=to_str, verbose=verbose)
            
        # keep track of feature usage in terms of thresholds at splitting points
        counts = count_features2(model, feature_names=feature_set, counts=counts, labels=labels, verbose=True)
        # ... counts: feature -> list of thresholds (used to estimate its median across decision paths)

        # visualization?
    ### end foreach trial 
    print("\n(analyze_path) Averaged AUC: {} | n_trials={}".format(np.mean(auc_scores), n_trials))
            
    return paths, counts

def load_data(input_file, **kargs): 
    """

    Memo
    ----
    1. Example datasets

        a. multivariate imputation applied 
            exposures-4yrs-merged-imputed.csv
        b. rows with 'nan' dropped 
            exposures-4yrs-merged.csv

    """
    import collections
    import data_processor as dproc

    X, y, features = dproc.load_data(input_path=dataDir, input_file=input_file, sep=',') # other params: input_path/None

    print("(load_data) dim(X): {}, sample_size: {}".format(X.shape, X.shape[0]))

    counts = collections.Counter(y)
    print("... class distribution | classes: {} | sizes: {}".format(list(counts.keys()), counts))
    print("... variables: {}".format(features))

    return X, y, features


def runWorkflow2(X, y, features, model=None, param_grid={}, **kargs):
    # 2. define model (e.g. decision tree)
    if verbose: print("(runWorkflow) 2. Define model (e.g. decision tree and its parameters) ...")
    ######################################################
    
    ######################################################
    # ... min_samples_leaf: the minimum number of samples required to be at a leaf node. 
    #     A split point at any depth will only be considered if it leaves at least 
    #     min_samples_leaf training samples in each of the left and right branches
    counts = {}

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
                            merge_labels=False, policy_count='standard', experiment_id=file_prefix,
                               create_dir=True, index=0, validate_tree=False, to_str=True, verbose=False)

        summarize_paths(paths, topk=topk)

        # for k, ths in counts.items(): 
        #     assert isinstance(ths, list), "{} -> {}".format(k, ths)
    if counts: 
        msg = "(runWorkflow2) Count feature usage (applicable for tree-based methods e.g. decision tree) ...\n"
        fcounts = [(k, len(ths)) for k, ths in counts.items()]
        sorted_features = sorted(fcounts, key=lambda x: x[1], reverse=True)
        msg += "> Top {} features:\n{}\n".format(topk_vars, sorted_features[:topk_vars])
        print(msg)

    return model

def runWorkflow(**kargs):
    def summarize_paths(paths, topk=10):
        labels = np.unique(list(paths.keys()))
    
        print("\n> 1. Frequent decision paths by labels (n={})".format(len(labels)))
        sorted_paths = sort_path(paths, labels=labels, merge_labels=False, verbose=True)
        for label in labels: 
            print("... Top {} paths (@label={}):\n{}\n".format(topk, label, sorted_paths[label][:topk]))
            
        print("> 2. Frequent decision paths overall ...")
        sorted_paths = sort_path(paths, labels=labels, merge_labels=True, verbose=True)
        print("> Top {} paths (overall):\n{}\n".format(topk, sorted_paths[:topk]))

        return
    def summarize_vars(X, y): 
        counts = collections.Counter(y)
        print("... class distribution | classes: {} | sizes: {}".format(list(counts.keys()), counts))

    from data_processor import load_data
    from utils_tree import visualize, sort_path
    import operator
    
    verbose = kargs.get('verbose', True)

    # 1. define input dataset 
    if verbose: print("(runWorkflow) 1. Specifying input data ...")
    ######################################################
    input_file = 'exposures-4yrs-merged-imputed.csv'
    file_prefix = input_file.split('.')[0]
    ######################################################

    X, y, features = load_data(input_path=dataDir, input_file=input_file, exclude_vars=['Gender', 'Zip Code'], verbose=True)

    # 2. define model (e.g. decision tree)
    if verbose: print("(runWorkflow) 2. Define model (e.g. decision tree and its parameters) ...")
    ######################################################
    p_grid = {"max_depth": [3, 4, 5, 8, 10, 15], 
              "min_samples_leaf": [1, 5, 10, 15, 20]}
    ######################################################
    # ... min_samples_leaf: the minimum number of samples required to be at a leaf node. 
    #     A split point at any depth will only be considered if it leaves at least 
    #     min_samples_leaf training samples in each of the left and right branches

    model = DecisionTreeClassifier(criterion='entropy', random_state=1)

    # 3. visualize the tree (deferred to analyze_path())
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
         analyze_path(X, y, model=model, p_grid=p_grid, feature_set=features, n_trials=100, n_trials_ms=30, save=False,  
                        merge_labels=False, policy_count='standard', experiment_id=file_prefix,
                           create_dir=True, index=0, validate_tree=False, to_str=True, verbose=False)

    summarize_paths(paths, topk=topk)

    # for k, ths in counts.items(): 
    #     assert isinstance(ths, list), "{} -> {}".format(k, ths)
    fcounts = [(k, len(ths)) for k, ths in counts.items()]
    sorted_features = sorted(fcounts, key=lambda x: x[1], reverse=True)
    print("> Top {} features:\n{}\n".format(topk_vars, sorted_features[:topk_vars]))
    

    return

if __name__ == "__main__":
    runWorkflow()