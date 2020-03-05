import os, sys
import collections, random

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor    # Import decision tree classifier model
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

import numpy as np
import pandas as pd
import common

from tabulate import tabulate
from utils_plot import saveFig

from analyzer import run_model_selection
from tree_analyzer import analyze_path

from loinc import FeatureSet, MatchmakerFeatureSet
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

def apply_model_to_predict(X, y=[], p_th=-1, model=None, params={}, **kargs): 
    
    verbose = kargs.get('verbose', 1)

    assert isinstance(params, dict)
    if model is None: 
        info_gain_measure = kargs.get('criterion', 'entropy')
        random_state = kargs.get('random_state', 53)
        model = DecisionTreeClassifier(criterion=info_gain_measure, random_state=random_state)

    pos_label = MatchmakerFeatureSet.pos_label
    neg_label = MatchmakerFeatureSet.neg_label

    y_pred = model.predict_proba(X)[:, 1]

    # label prediction
    if p_th > 0: 
        y_label = np.repeat(neg_label, X.shape[0])
        pos_index = np.where(y_pred >= p_th)[0]
        neg_index = np.where(y_pred < p_th)[0]

        assert len(set(pos_index).intersection(neg_index)) == 0

        y_label[pos_index] = pos_label
        y_label[neg_index] = neg_label

        if verbose: 
            print("[predict] Found n={} predicted positive, m={} negative | p_th={}".format(len(pos_index), len(neg_index), p_th))
    else: 
        y_label = model.predict(X)

    # [test]
    # if true labels 'y' is given
        
    return y_pred, y_label

def analyzeDecisionPaths(X, y, features=[], **kargs):
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

    from data_processor import load_generic, toXY
    from utils_tree import visualize, sort_path
    import operator, time
    
    verbose = kargs.get('verbose', True)
    if len(features) == 0: features = [f"x{i}" for i in range(X.shape[1])]

    # experiment params 
    validate_ms = kargs.get("validate_ms", True) 
    # if True, a plot will be generated that ... 
    # ... compares the model selection in the setting of nested vs non-nested CV 
    nruns_ms = kargs.get("nruns_ms", 10) # number of iterations for model selection
    n_trials = kargs.get("n_trials", 30) # number of iterations for running a post-model-selection model
    experiment_id = kargs.get("experiment_id", "test")

    # DT params 
    # treeviz_name = kargs.get('treeviz_name', 'test') # output of the DT visualizations
    validate_tree = kargs.get('validate_tree', True)
    plot_ext = kargs.get("plot_ext", "tif")

    # Define model (e.g. decision tree)
    if verbose: print("(runWorkflow) 2. Define model (e.g. decision tree and its parameters) ...")
    ######################################################
    p_grid = {"max_depth": [3, 4, 5, 8, 10, 15], 
              "min_samples_leaf": [1, 5, 10, 15, 20]}
    ######################################################
    # ... min_samples_leaf: the minimum number of samples required to be at a leaf node. 
    #     A split point at any depth will only be considered if it leaves at least 
    #     min_samples_leaf training samples in each of the left and right branches

    model = DecisionTreeClassifier(criterion='entropy', random_state=int(time.time()))
    best_params, params, nested_scores = \
            run_model_selection(X, y, model, p_grid=p_grid, n_runs=nruns_ms, 
                output_path="", meta=experiment_id, ext=plot_ext, save=validate_ms)
    print('(analyzeDecisionPaths) best_params(DT):\n{}\n'.format(best_params))

    ###################################################### 
    test_size = 0.3
    rs = 53
    topk = 10
    topk_vars = 10
    ######################################################
    labels = [str(l) for l in sorted(np.unique(y))]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs) # 70% training and 30% test

    # params = {'max_depth': 5}  # use model selection to determine the optimal 'max_depth'
    # model = apply_model(X_train, y_train, params=params, random_state=rs)
    # graph = visualize(model, features, labels=labels, plot_dir=plotDir, file_name=file_prefix, ext='tif')

    # 4. analyze decision paths and keep track of frequent features
    paths, counts, scores = \
        analyze_path(X, y, feature_set=features, model=model, best_params=best_params, n_trials=n_trials, n_trials_ms=nruns_ms,  
                merge_labels=False, policy_count='standard', experiment_id=experiment_id,
                    create_dir=True, index=0, 
                        validate_tree=validate_tree, plot_ext=plot_ext,
                            to_str=True, 
                            verbose=verbose,
                            return_meta_data=True) # return meta data to include performance scores and probability thresholds during training
    
    best_thresholds = scores['p_threshold']
    best_scores = scores['fmax']

    summarize_paths(paths, topk=topk)

    # for k, ths in counts.items(): 
    #     assert isinstance(ths, list), "{} -> {}".format(k, ths)
    fcounts = [(k, len(ths)) for k, ths in counts.items()]
    sorted_features = sorted(fcounts, key=lambda x: x[1], reverse=True)
    print("> Top {} features:\n{}\n".format(topk_vars, sorted_features[:topk_vars]))
    
    return model, best_params, best_scores, best_thresholds

def demo_predict(**kargs): 
    from data_processor import load_generic, toXY
    import feature_gen as fg

    col_target = MatchmakerFeatureSet.col_target
    col_assignment = MatchmakerFeatureSet.col_assignment
    col_score = MatchmakerFeatureSet.col_score

    # load data
    cohort = kargs.get('cohort', 'hepatitis-c')
    verbose = kargs.get('verbose', 1)
    scaling_method = kargs.get('standardize')
    n_trials = kargs.get("n_trials", 100)
   
    # control logic 
    include_meta = kargs.get("include_meta", True)

    # conf_threshold = 0.8  # if the matchmaker produces a probability score (of a match) less than this number, ... 
    # ... then we suspect that this LOINC assignment is probably wrong, and we then need to seek other candidates

    # 1. define input dataset 
    if verbose: print("(main) 1. Specifying input data ...")

    # load data and remove meta variables
    # -----------------------------------------------------
    ts_train = fg.load_dataset(dtype='train', include_meta=include_meta)
    print("... dim(ts_train): {}".format(ts_train.shape))

    matching_vars, regular_vars, target_vars, derived_vars, meta_vars = \
        MatchmakerFeatureSet.categorize_transformed_features(ts_train, remove_prefix=False)
    assert len(matching_vars) > 0
    if include_meta: assert len(meta_vars) > 0
    dfM, dfX, dfY, df_derived, df_meta = MatchmakerFeatureSet.partition(ts_train, verbose=1, mode='posterior')
    assert dfM.shape[1] == len(matching_vars) and dfX.shape[1] == len(regular_vars), \
        "ncols(dfM)={} ~? {}, ncols(dfX)={} ~? {}".format(dfM.shape[1], len(matching_vars), dfX.shape[1], len(regular_vars))
    ts_train = pd.concat([dfM, dfX, dfY], axis=1)
    # ------------------------------------------------------

    X, y, features, labels = toXY(ts_train, cols_y=[col_target, ], scaler=scaling_method, pertube=False)
    y = y.flatten()

    model, best_params, best_scores, best_thresholds = \
        analyzeDecisionPaths(X, y, features=features, 
            validate_tree=True, validate_ms=True, n_trials=n_trials,
                experiment_id=f'matchmaker-tree-{cohort}', plot_ext='tif') 
    # ... 'pdf' doesn't seem to work

    #########################################################
    # test data

    # -------------------------------------------------------
    ts_test = fg.load_dataset(dtype='test', include_meta=True, cohort=cohort, drop_untracked=True, save_result=True)
    # ... remember to set drop_untracked to True in order to drop previous experimental results 

    dfM, dfX, dfY, df_derived, df_meta = MatchmakerFeatureSet.partition(ts_test, verbose=1, mode='posterior')
    assert dfM.shape[1] == len(matching_vars) and dfX.shape[1] == len(regular_vars), \
        "ncols(dfM)={} ~? {}, ncols(dfX)={} ~? {}".format(dfM.shape[1], len(matching_vars), dfX.shape[1], len(regular_vars))
    ts_test = pd.concat([dfM, dfX, dfY], axis=1)
    # -------------------------------------------------------

    # test data may not have 'match_status' attribute 
    X_test, y_test, _, _ = toXY(ts_test, cols_y=[col_target, ], scaler=scaling_method, perturb=False)
    print("... dim(X): {}, dim(X_test): {}".format(X.shape, X_test.shape))

    # -------------------------------------------------------
    print("(demo) Best prob thresholds:\n{}\n".format(best_thresholds))
    th_scores = sorted(zip(best_thresholds, best_scores), key=lambda x: x[1], reverse=True)
    print("(demo) sorted threshold vs scores:\n{}\n".format(th_scores))
    best_threshold = np.mean([pth for pth, score in th_scores[:5]])
    print("...    choose p_threshold: {}".format(best_threshold))
    # best_threshold = np.mean(best_thresholds[0])
    # -------------------------------------------------------

    y_pred, y_label = apply_model_to_predict(X_test, y=[], model=model, params=best_params, p_th=-1)
    print("... y_pred:\n{}\n".format(y_pred))

    hit_index = np.where(y_label==1)[0] # for these rows, the predition is "correct";therefore, the predition is the same as the one already documented
    miss_index = np.where(y_label==0)[0] # for these rows, we need to scan through another set of candidates

    ts_test[col_score] = y_pred
    ts_test[col_target] = y_label

    if col_score in df_meta.columns: df_meta.drop([col_score], axis=1, inplace=True)
    ts_test = pd.concat([ts_test, df_derived, df_meta], axis=1)

    ts_test.sort_values(by=[col_target, col_score], ascending=False, inplace=True)
    fg.save_dataset(ts_test, dtype='test', cohort=cohort, verbose=1, suffix=0)
    # ... append a 'suffix' to not overwrite the original data
    ###########################################################
    # ... Save analysis data

    header = ["existing", "confidence", "correctness", "alternatives"]
    predictions = {h:[] for h in header}

    
    # propose candidates
    # rank candidates    

    return

def test(**kargs):

    demo_predict()

    return 

if __name__ == "__main__":
    test()