import pandas as pd 
import numpy as np
from pandas import DataFrame, Series
import os

from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor    # Import decision tree classifier model
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from tabulate import tabulate
import common

from utils_plot import saveFig # contains "matplotlib.use('Agg')" which needs to be called before pyplot 
from matplotlib import pyplot as plt

"""


Update
------
    1. Factor codes from exposure_analyzer.py (from asthma_env module) to this module   ... 12.17.19

"""
dataDir = os.path.join(os.getcwd(), 'data')  # default unless specified otherwise
plotDir = os.path.join(os.getcwd(), 'plot')

class Data(object): 
    label = 'label'
    features = []

#################################################################################
# Utility Function 

def summarize_loinc(codes, df=None, col="test_result_loinc_code", n=10, codebook={}):
    if n > 0: codes = np.random.choice(codes, n)
    if codebook: 
        pass
    else: 
        if df is None: 
            for code in codes: 
                print("[%s]" % code)
        else: 
            # for r, dfg in df.groupby([col,]): 
            for code in codes: 
                dfc = df.loc[df[col] == code]
                print("[{}] n={}".format(code, dfc.shape[0]))
    return
       
def col_values(df, col='age', n=10):
    df_subset = df.sample(n=n, random_state=1) # for each column, sample n values (usually n=1)
    return df_subset[col].values
sample_col_values = col_values

def interpret(df, n=1, verbose=2, save_uniq_vals=True, output_dir='', sep=','): 
    if df.empty:
        print("(interpret) Empty dataframe!")
    
    assert n==1, "Not supported. Coming soon :)"
    if not output_dir: output_dir = os.getcwd()
    
    # df_subset = df.sample(n=n, random_state=1) # for each column, sample n values (usually n=1)
    
    # dtype_lookup = dict(df_subset.dtypes) 
    # ...  because columns contain null values, dtype info is not useful if nulls are not removed
    N = df.shape[0]
    adict = {col:np.array([]) for col in df.columns} # map col -> unique values
    vcounts = {col:0 for col in df.columns}  # value counts
    for col in df.columns: 
        # df[col].notnull()
        vals = df.loc[~df[col].isnull()][col]
        adict[col] = vals.unique() 
        vcounts[col] = len(vals)
    # ... now we know non-null column values 
        
    # analyze the data 
    header = ['column', 'cardinality', 'dtype', 'values', 'r_missing']
    dvals = {h: [] for h in header}
    for k, v in adict.items():  # each col-values pair ~ one row in dvals
        # d_values[k] = ' '.join(v)
        
        # uv = np.unique(v) # [log] '<' not supported between instances of 'float' and 'str'
        uv = v 
        ratio_missing = round((N-vcounts[k])/(N+0.0), 4)
        
        dvals['column'].append(k) # column/attribute
        dvals['values'].append( uv ) # unique values
        dvals['cardinality'].append(len(uv))  # unique values will likely differ by columns
        dvals['r_missing'].append( ratio_missing )    # ratio of missing values
        dvals['dtype'].append( v.dtype.name )  # dtype after removing nulls
        
        msg = ''
        if verbose: 
            msg += "[{}] => n_uniq={}/(n={},N={} | r_miss={}%), dtype={}\n".format(
                k, len(uv), vcounts[k], N, ratio_missing*100, uv.dtype.name)
            if len(uv) > 0:
                msg += '  :{}\n'.format( list(np.random.choice(uv, min(n, len(uv)))) )
            print(msg)
        
    # save the value dataframe
    if save_uniq_vals: 
        file_uniq_vals = 'uniq_vals.csv'  # [todo]
        output_path = os.path.join(output_dir, file_uniq_vals)
        if verbose > 1: print('(interpret) Saving df(unique values) to {}'.format(output_path))
            
        # only show up to 3 decimal places for floats
        DataFrame(dvals).to_csv(output_path, sep=sep, index=False, header=True, float_format='%.3f') 

    return
        
        
def summarize_dataframe(df, n=1): 
    msg = ""
    msg += "> sample sizes: {}\n".format(df.shape[0])
    msg += "> n(features):  {}\n".format(df.shape[1])
    # msg += "> list of features:\n{}\n".format(df.columns.values)
    
    # using dataframe's utility 
    msg += "> df.describe():\n{}\n".format(df.describe())

    print(msg)

    interpret(df, n=n, verbose=True)
# alias 
summary = summarize_dataframe


# Classifier Utilties
#######################################################

def run_model_selection(X, y, model, p_grid={}, n_trials=30, scoring='roc_auc', output_path='', output_file='', create_dir=True, 
                        index=0, plot_=True, ext='tif', save=False, verbose=1): 
    from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
    from matplotlib import pyplot as plt
    
    # Arrays to store scores
    non_nested_scores = np.zeros(n_trials)
    nested_scores = np.zeros(n_trials)

    # Loop for each trial
    icv_num = 5
    ocv_num = 5
    best_params = {}
    for i in range(n_trials):
        if verbose: 
            print("(run_model_selection) Trial #{} ......".format(i+1))
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

def eval_performance(X, y, model=None, cv=5, random_state=53, **kargs):
    """

    Memo
    ----
    1. https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

    """

    # from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
    from sklearn.model_selection import train_test_split, StratifiedKFold # Import train_test_split function
    from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc, f1_score
    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
    from common import fmax_score_threshold
    import time
    
    #### parameters ####
    test_size = kargs.get('test_size', 0.2) # save
    verbose = kargs.get('verbose', 0)
    experiment_id = kargs.get('experiment_id', 'test') # a file ID for the output (e.g. example decision tree)
    plot_dir = kargs.get('plot_dir', 'plot')
    plot_ext = kargs.get('plot_ext', 'tif')
    ####################

    # clf = LogisticRegressionCV(cv=5, random_state=random_state, scoring=).fit(X, y)
    if not model: 
        model = LogisticRegression(class_weight='balanced')

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    cv_scores =[]
    for i, (train, test) in enumerate(kf.split(X,y)):
        if verbose: print('> {} of KFold {}'.format(i, kf.n_splits))
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        
        #model
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:,1]
        score, p_th = fmax_score_threshold(y_test, y_pred)

        # other metrics
        y_pred_label = model.predict(X_test)
        score_f1 = f1_score(y_test, y_pred_label)
        score_auc = roc_auc_score(y_test, y_pred)

        if verbose: 
            print('> Fmax: {} p_th: {} | F1: {}, AUC: {}'.format(score, p_th, score_f1, score_auc))
        
        cv_scores.append(score)    

    return cv_scores


def analyze_path(X, y, model=None, p_grid={}, feature_set=[], n_trials=100, n_trials_ms=10, 
                    save_ms=False, save_dt_vis=True, 
                        output_path='', output_file='', 
                             create_dir=True, index=0, **kargs):
    """

    Given the training data (X, y) and a tree-based model (e.g. Decision Tree), 
    train the model and analyze the decision paths: 
       1) count the occurrences of decision paths

    """
    # from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
    from sklearn.model_selection import train_test_split # Import train_test_split function
    from utils_tree import visualize, count_paths, count_paths2, count_features2
    import time
    
    #### parameters ####
    test_size = kargs.get('test_size', 0.2) # save
    verbose = kargs.get('verbose', 0)
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
    if verbose > 1: 
        msg += "(analyze_path) dim(X): {} | vars (n={}):\n...{}\n".format(X.shape, len(feature_set), feature_set)
        msg += "... class distribution: {}\n".format(collections.Counter(y))
    print(msg)

    # define model 
    if model is None: model = DecisionTreeClassifier(criterion='entropy', random_state=time.time())
    
    # run model selection 
    if len(p_grid) > 0: 
        best_params, nested_scores = \
            run_model_selection(X, y, model, p_grid=p_grid, n_trials=n_trials_ms, output_path=output_path, ext=plot_ext, save=save_ms)
        the_index = np.argmax(nested_scores)
        the_params = best_params[np.argmax(nested_scores)]
        # print('> type(best_params): {}:\n{}\n'.format(type(best_params), best_params))
    print("(analyze_path) Model selection complete > best params: {}  #".format(the_params))
        
    # initiate data structures 
    paths = {}
    lookup = {}
    counts = {f: [] for f in feature_set} # maps features to lists of thresholds
        
    # build N different decision trees and compute their statistics (e.g. performance measures, decision path counts)
    metric_opt = ['auc', 'fmax', 'accuracy']
    scores = {m: [] for m in metric_opt}
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
                file_prefix = "{id}-{index}".format(id=experiment_id, index=i)
                graph = visualize(model, feature_set, labels, file_name=file_prefix, ext='tif', save=save_dt_vis)
                
                # display the tree in the notebook
                # Image(graph.create_png())  # from IPython.display import Image
        
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        auc_score = metrics.roc_auc_score(y_test, y_pred)
        fmax_score = common.fmax_score(y_test, y_pred, beta = 1.0, pos_label = 1)
        fmax_score_negative = common.fmax_score(y_test, y_pred, beta = 1.0, pos_label = 0)
        if i % 10 == 0: 
            print("[{}] Accuracy: {}, AUC: {}, Fmax: {}, Fmax(-): {}".format(i, accuracy, auc_score, fmax_score, fmax_score_negative))
        scores['auc'].append(auc_score)
        scores['fmax'].append(fmax_score)
        scores['accuracy'].append(accuracy)

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
    print("\n(analyze_path) E[acc]: {}, E[AUC]: {}, E[fmax]: {} | n_trials={}".format(
        np.mean(scores['accuracy']), np.mean(scores['auc']), np.mean(scores['fmax']), n_trials))
            
    return paths, counts

def load_data0(input_file, **kargs): 
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

### I/O Utilities ###

def save_generic(df, cohort='', dtype='ts', output_file='', sep=',', **kargs):
    output_dir = kargs.get('output_dir', os.path.join(os.getcwd(), 'data')) 
    verbose=kargs.get('verbose', 1)
    if not output_file: 
        if cohort: 
            output_file = f"{dtype}-{cohort}.csv" 
        else: 
            output_file = "test.csv"
    output_path = os.path.join(output_dir, output_file)

    df.to_csv(output_path, sep=sep, index=False, header=True)
    if verbose: print("(save_generic) Saved dataframe (dim={}) to:\n{}\n".format(df.shape, output_path))
    return  
def load_generic(cohort='', dtype='ts', input_file='', sep=',', **kargs):
    input_dir = kargs.get('input_dir', os.path.join(os.getcwd(), 'data'))

    warn_bad_lines = kargs.get('warn_bad_lines', True)
    verbose=kargs.get('verbose', 1)
    columns = kargs.get('columns', [])

    if not input_file: 
        if cohort: 
            input_file = f"{dtype}-{cohort}.csv" 
        else: 
            input_file = "test.csv"
    input_path = os.path.join(input_dir, input_file)

    if os.path.exists(input_path) and os.path.getsize(input_path) > 0: 
        df = pd.read_csv(input_path, sep=sep, header=0, index_col=None, error_bad_lines=False, warn_bad_lines=warn_bad_lines)
        if verbose: print("(load_generic) Loaded dataframe (dim={}) from:\n{}\n".format(df.shape, input_path))
    else: 
        df = None
        if verbose: print("(load_generic) No data found at:\n{}\n".format(input_path))

    if len(columns) > 0: 
        return df[columns]
    return df  

def save_data(df, cohort='', output_file='', sep=',', **kargs): 
    output_dir = kargs.get('output_dir', os.path.join(os.getcwd(), 'data'))
    verbose=kargs.get('verbose', 1)
    if not output_file: 
        if cohort: 
            output_file = f"ts-{cohort}.csv" 
        else: 
            output_file = "ts-generic.csv"
    output_path = os.path.join(output_dir, output_file)

    df.to_csv(output_path, sep=sep, index=False, header=True)
    if verbose: print("(save_data) Saved dataframe (dim={}) to:\n{}\n".format(df.shape, output_path))
    return
def load_data(cohort='', input_file='', sep=',', **kargs):
    input_dir = kargs.get('input_dir', os.path.join(os.getcwd(), 'data'))
    
    warn_bad_lines = kargs.get('warn_bad_lines', True)
    verbose=kargs.get('verbose', 1)
    columns = kargs.get('columns', [])
    if not input_file: 
        if cohort: 
            input_file = f"ts-{cohort}.csv" 
        else: 
            input_file = "ts-generic.csv"
    input_path = os.path.join(input_dir, input_file)

    df = pd.read_csv(input_path, sep=sep, header=0, index_col=None, error_bad_lines=False, warn_bad_lines=warn_bad_lines)
    if verbose: print("(load_data) Loaded dataframe (dim={}) from:\n{}\n".format(df.shape, input_path))
    
    if len(columns) > 0: 
        return df[columns]
    return df
def load_data_incr(cohort='', input_file='', sep=',', **kargs): 
    input_dir = kargs.get('input_dir', os.path.join(os.getcwd(), 'data'))
    
    warn_bad_lines = kargs.get('warn_bad_lines', True)
    verbose=kargs.get('verbose', 1)
    columns = kargs.get('columns', [])
    chunksize = kargs.get('chunksize', 1000000)

    if not input_file: 
        if cohort: 
            input_file = f"ts-{cohort}.csv" 
        else: 
            input_file = "ts-generic.csv"
    input_path = os.path.join(input_dir, input_file)

    for dfi in pd.read_csv(input_path, sep=sep, header=0, 
        index_col=None, error_bad_lines=False, warn_bad_lines=warn_bad_lines, chunksize=chunksize): 
        if len(columns) > 0: 
            yield dfi[columns]
        else: 
            yield dfi

def load_src_data(cohort='hepatitis-c', **kargs): 
    input_dir = kargs.get('input_dir', os.path.join(os.getcwd(), 'data')) 
    input_file = f"andromeda-pond-{cohort}.csv" # "andromeda_pond-10p.csv"
    input_path = os.path.join(input_dir, input_file)

    warn_bad_lines = kargs.get('warn_bad_lines', True)
    columns = kargs.get('columns', [])
    df = pd.read_csv(input_path, sep=',', header=0, index_col=None, error_bad_lines=False, warn_bad_lines=warn_bad_lines)

    if len(columns) > 0: 
        return df[columns]
    return df
    
def save_performnace(df, output_dir='result', output_file='', **kargs): 
    cohort = kargs.get('cohort', 'hepatitis-c')
    sep = kargs.get('sep', '|')
    verbose = kargs.get('verbose', 1)

    output_dir = kargs.get('output_dir', os.path.join(os.getcwd(), output_dir)) 
    output_file = f"performance-{cohort}.csv" 
    output_path = os.path.join(output_dir, output_file)
    df.to_csv(output_path, sep=sep, index=False, header=True)

    if verbose: 
        print('(save) Saving performance dataframe to:\n{}\n ... #'.format(output_path))
        for code, score in zip(df_perf['code'], df_perf['mean']):
            print(f"[{code}] -> {score}")
    return

def load_performance(input_dir='result', input_file='', **kargs):
    cohort = kargs.get('cohort', 'hepatitis-c')
    sep = kargs.get('sep', '|')

    if not input_dir: input_dir = kargs.get('input_dir', 'result') # os.path.join(os.getcwd(), 'result')
    if not input_file: input_file = f"performance-{cohort}.csv" 
    input_path = os.path.join(input_dir, input_file)
    assert os.path.exists(input_path), "Invalid path: {}".format(input_path)

    df = pd.read_csv(input_path, sep=sep, header=0, index_col=None, error_bad_lines=False)
    print("> dim(performance matrix): {}".format(df.shape)) 

    return df

def label_by_performance(cohort='hepatitis-c', th_low =0.50, th_high = 0.90, categories=[], verify=True, verbose=1): 
    """
    Classify loinc codes by their performance scores. 
    """
    from loinc import is_valid_loinc
    df_perf = load_performance(input_dir='result', cohort=cohort)
    target_codes = codes = df_perf['code'].values  # codes associated with a given cohort (e.g. hepatitis c)
    n_codes = len(codes)

    if verify: 
        verbose += 1
        for c in np.random.choice(target_codes, 100): 
            assert is_valid_loinc(c, token_default='unknown', dehyphenated=True)

    codes_low_sz = df_perf.loc[df_perf['mean'] < 0]['code'].values
    codes_scored = df_perf.loc[df_perf['mean'] >= 0]['code'].values # <<<<< 
    codes_high_score = df_perf.loc[df_perf['mean'] >= th_high]['code'].values
    codes_low_score = df_perf.loc[(df_perf['mean'] < th_high) & (df_perf['mean'] >= 0)]['code'].values
    r_low_sz = round( len(codes_low_sz)/(n_codes+0.0) * 100, 3)
    r_scored = round( len(codes_scored)/(n_codes+0.0) * 100, 3)
    r_high_score = round( len(codes_high_score)/(n_codes+0.0) * 100, 3)

    if verbose: 
        print("(label_by_performance) dim(df_perf): {}".format(df_perf.shape))
        print("1. Total number of codes: {} | n(low_sample): {}({}%), n(scored):{}({}%), n(high scored):{}({}%)".format(n_codes, 
            len(codes_low_sz), r_low_sz, len(codes_scored), r_scored, len(codes_high_score), r_high_score))
        print("2. Pecentage scored codes: {}%".format(r_scored))
        print("3. Percentage \"good\" codes: {}%".format(r_high_score))

    ccmap = {}  # ccmap: code to category
    ccmap['easy'] = ccmap[0] = codes_high_score
    ccmap['hard'] = ccmap[1] = codes_low_score
    ccmap['low'] = ccmap[2] = codes_low_sz

    # focus only these categories if provided
    if len(categories) > 0: 
        ccmap = {cat: ccmap[cat] for cat in categories}

    return ccmap

def load_loinc_table(input_dir='LoincTable', input_file='', **kargs):
    from transformer import dehyphenate
    import loinc as ul

    sep = kargs.get('sep', ',')
    dehyphen = kargs.get('dehyphenate', False)

    if not input_dir: input_dir = kargs.get('input_dir', "LoincTable") # os.path.join(os.getcwd(), 'result')
    if not input_file: input_file = "Loinc.csv"
    input_path = os.path.join(input_dir, input_file)
    assert os.path.exists(input_path), "Invalid path: {}".format(input_path)

    df = pd.read_csv(input_path, sep=sep, header=0, index_col=None, error_bad_lines=False)
    print("> dim(table): {}".format(df.shape)) 

    if dehyphen: 
        col_key = ul.LoincTable.table_key_map.get(input_file, 'LOINC_NUM')
        df = dehyphenate(df, col=col_key)  # inplace

    return df

def load_loinc_to_mtrt(input_file='loinc-leela.csv', **kargs):
    sep = kargs.get('sep', ',')
    input_dir = kargs.get('input_dir', os.path.join(os.getcwd(), 'data'))
    df = load_generic(input_dir=input_dir, input_file=input_file, sep=sep) 

    return df

def sample_df_values(df, cols=[], **kargs): 

    n_samples = kargs.get('n_samples', 10) # show example n values
    verbose = kargs.get('verbose', 1)

    if not cols: cols = df.columns.values

    msg = ''
    N = df.shape[0] 
    df = df.sample(n=min(N, n_samples))

    adict = {}
    for i, col in enumerate(cols): 
        ieff = i+1
        msg += f"[{ieff}] (n={n_samples}):\n"

        adict[col] = list(df[col].values)
        msg += "    + {}: {}\n".format(col, list(df[col].values))
    if verbose: print(msg)
    return adict

def list_unique_values(df, cols=[], **kargs):
    verbose = kargs.get('verbose', 1)
    n_max = kargs.get('n_max', 50) # show at most this number of values 
    if not cols: cols = df.columns.values

    # adict = sample_df_values(df, cols=cols, n_samples=df.shape[0])
    msg = ''
    adict = {}
    for i, col in enumerate(cols): 
        ieff = i+1
        adict[col] = list(df[col].unique())
        msg += "[{}] {} (n={}) => {}\n".format(ieff, col, len(adict[col]), adict[col])
    if verbose: print(msg)
        
    return adict

def det_cardinality(df, **kargs):
    verbose = kargs.get('verbose', 1)
    th_card = kargs.get('th_card', 10) # if n(uniq values) >= this number => high card
    target_cols = kargs.get('cols', [])
    if not target_cols: target_cols = df.columns.values
    # print("(det_cardinality) target cols: {}".format(target_cols))

    adict = list_unique_values(df, cols=target_cols, verbose=verbose) 
    
    if th_card > 0: 
        for col, values in adict.items(): 
            if col in target_cols: 
                if th_card > 0: 
                    adict[col] = int(len(np.unique(values)) >= th_card) 
                else: 
                    adict[col] = len(np.unique(values))
    return adict

def sample_loinc_table(codes=[], cols=[], input_dir='LoincTable', input_file='', **kargs): 
    from transformer import dehyphenate
    # from tabulate import tabulate

    col_key = kargs.get('col_key', 'LOINC_NUM')
    n_samples = kargs.get('n_samples', 10) # if -1, show all codes
    verbose = kargs.get('verbose', 1)

    df = load_loinc_table(input_dir=input_dir, input_file=input_file, **kargs)
    df = dehyphenate(df, col=col_key)  # inplace

    if not cols: cols = df.columns.values
    if len(codes) == 0: 
        codes = df.sample(n=n_samples)[col_key].values
    else:  
        codes = np.random.choice(codes, min(n_samples, len(codes)))

    msg = ''
    code_documented = set(df[col_key].values) # note that externally provided codes may not even be in the table!
    adict = {code:{} for code in codes}
    for i, code in enumerate(codes):  # foreach target codes
        ieff = i+1
        if code in code_documented: 
            dfi = df.loc[df[col_key] == code] # there should be only one row for a given code
            assert dfi.shape[0] == 1, "code {} has multiple rows: {}".format(code, tabulate(dfi, headers='keys', tablefmt='psql'))
            msg += f"[{ieff}] loinc: {code}:\n"
            for col in cols: 
                v = list(dfi[col].values)
                if len(v) == 1: v = v[0]
                msg += "  - {}: {}\n".format(col, v)

            adict[code] = sample_df_values(dfi, verbose=0) # sample_df_values() returns a dictionary: column -> value
    if verbose: print(msg)

    return adict

def analyze_by_values(ccmap, cols=[], verbose=1, col_code=''): 
    """
    Input
    -----
    ccmap: a dictionary from 'class' to a list of codes associated with the class
           where a class given to each LOINC code to specify an analysis category (e.g. easy vs hard, 
           as in easy to classify or hard to classify)

           Use label_by_* methods to generate this ccmap (i.e. code-to-category mapping)
           (e.g. label_by_performance()) 
    
    """
    import loinc as ul   
    from transformer import canonicalize, dehyphenate
    assert isinstance(ccmap, dict)

    df_loinc = load_loinc_table(dehyphenate=True) # input_dir/'LoincTable', input_file/'LoincTable.csv', sep/','
    # dehyphenate(df_loinc, col=col_code)  # inplace

    if not col_code: col_code = ul.LoincTable.col_key  # 'LOINC_NUM'
    

    target_cols = cols
    if len(cols) == 0: 
        
        text_cols = ul.LoincTable.text_cols #  ['LONG_COMMON_NAME', 'SHORTNAME', 'RELATEDNAMES2', 'STATUS_TEXT']
        property_cols = ul.LoincTable.p6 + ['CLASS', ] # [ 'COMPONENT', 'PROPERTY', 'TIME_ASPCT', 'SYSTEM', 'METHOD_TYP', 'SCALE_TYP', 'CLASS', ]  
        target_cols = [col_code, ] + text_cols + property_cols
    # ... only select above columns from the loinc table

    D = {}
    for i, (cat, codes) in enumerate(ccmap.items()): 
        assert len(codes) > 0
        # e.g. codes_hard = df.loc[df['label'] == 'hard'][ 'code' ].values

        # if i < 10: 
        #     print("> cat: {} => codes: {}".format(cat, codes[:10]))
        #     print("> codes_src: {}".format(df_loinc[col_code]))
        
        # take the rows of these codes and inspect the columns in 'target_cols'
        D[cat] = df_loinc.loc[df_loinc[col_code].isin(codes)][target_cols]

        if verbose: print("(analyze_by_values) Case {} (n={}):".format(cat, len(codes)))
        for i, (r, row) in enumerate(D[cat].iterrows()):
            ieff = i+1
            code = row[col_code]
            p6 = [row['PROPERTY'], row['TIME_ASPCT'], row['SYSTEM'], row['METHOD_TYP'], row['SCALE_TYP'], row['CLASS']]
            six_parts = ', '.join(str(e) for e in p6)
            if verbose: print("  [{}] {} (6p: <{}>) =>\n ... {}\n".format(ieff, code, six_parts, row['LONG_COMMON_NAME']))

    return D

def runWorkflow(X, y, features, model=None, param_grid={}, **kargs):
    """

    Memo
    ----
    1. This is a modified version of exposure_analyzer.runWorkflow() to allow for arbitrary (X, y)
    """
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

    verbose = kargs.get('verbose', 1)
    file_prefix = kargs.get('experiment_id', 'loinc_predict_v0')
    n_runs_ms = kargs.get('n_runs_ms', 10)
    n_trials = kargs.get('n_trials', 10)
    save_plot = kargs.get('save_plot', True)
    ######################################################
    
    counts = {}

    # Define model (e.g. decision tree)
    if not model: 
        if verbose: print("(runWorkflow) Define model (e.g. decision tree and its parameters) ...")
        if not param_grid:  # p_grid
            ######################################################
            # ... min_samples_leaf: the minimum number of samples required to be at a leaf node. 
            #     A split point at any depth will only be considered if it leaves at least 
            #     min_samples_leaf training samples in each of the left and right branches
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
             analyze_path(X, y, model=model, p_grid=param_grid, feature_set=features, n_trials=n_trials, n_trials_ms=n_runs_ms, 
                            save_ms=False, save_dt_vis=True,  
                            merge_labels=False, policy_count='standard', experiment_id=file_prefix,
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

def one_vs_all_encoding(df, target_label, codebook={'pos': 1, 'neg': 0}, col='test_result_loinc_code', col_target='target'): 
    # inplace operation
    
    if isinstance(df, DataFrame): 
        assert col in df.columns 
        cond_pos = df[col] == target_label  # target loinc
        cond_neg = df[col] != target_label
        print("> target: {} (dtype: {}) | n(pos): {}, n(neg): {}".format(target, type(target), np.sum(cond_pos), np.sum(cond_neg)))
        df[col_target] = df[col]
        df.loc[cond_pos, col_target] = codebook['pos']
        df.loc[cond_neg, col_target] = codebook['neg'] 
    else: 
        # df is a numpy 1D array
        # print("> y: {}, target_label: {}".format(df, target_label))
        # print("> y': {}".format(np.where(df == target_label, codebook['pos'], codebook['neg'])))
        df = np.where(df == target_label, codebook['pos'], codebook['neg'])

    return df

def get_eff_values(df, col=''):
    if isinstance(df, DataFrame):
        assert col in df.columns
        return list(df[df[col].notnull()][col].values)
    else: 
        # df is a numpy array
        assert isinstance(df, np.ndarray)
        return list(df[~np.isnan(df)])

def get_sample_sizes(y, sorted_=True, col='test_result_loinc_code'): 
    import collections
    
    if isinstance(y, DataFrame): 
        # ulabels = df[col].unique()
        # print("... ulabels: {}".format(ulabels))

        sizes = collections.Counter( y[col].values )
        # for label in ulabels: 
        #    dfc = df.loc[df[col] == label]
        #    sizes[label] = dfc.shape[0]

        # if sorted_: 
            # sort by sample sizes (values)
        #    sizes = collections.OrderedDict( sorted(sizes.items(), key=operator.itemgetter(1), reverse=True) )
    else: 
        # df is a numpy array or list
        sizes = collections.Counter(y)
        
    return sizes # label/col -> sample size

def summarize_dict(d, topn=15, sort_=True): 
    if topn != 0 or sort_: 
        import operator
        d = sorted(d.items(), key=operator.itemgetter(1))
    for k, v in d[:topn]: 
        print(f"[{k}] -> {v}")
    return

def encode_labels(df, pos_label, neg_label=None, col_label='test_result_loinc_code', codebook={}, verbose=1): 
    if not codebook: codebook = {'pos': 1, 'neg': 0, '+': 1, '-': 0}
        
    if verbose: 
        # get non-null values of an attribute 
        # values = get_eff_values(df, col=col_label)
        # print("> all possible values:\n> {} ...".format(np.random.choice(values, 50)))
        # [log] strange values: 'UNLOINC', 'ALT (SGPT) P5P'
        pass
    
    y = df[col_label] if isinstance(df, DataFrame) else df
    sizes = get_sample_sizes(y)
    n0 = sizes[pos_label]
        # df.loc[df[col_target] == pos_label]

    col_target='target'
    y = one_vs_all_encoding(y, target_label=pos_label, codebook=codebook)
    # ... if df is a DataFrame, then df has an additional attribute specified by col_target/'target'

    sizes = get_sample_sizes(y)
    assert sizes[codebook['pos']] == n0

    print("(encode_labels) sample size: {}".format(sizes))
    
    return y
    
########################################
# Data utilities

def analyze_values(df, cols=[], verbose=1): 
    if not cols: cols = df.columns.values 

    adict = {}
    for i, col in enumerate(cols): 
        mv = collections.Counter(df[col]).most_common(10)
        mvz = [e[0] for e in mv]
        m = mv[0]
        mn, mc = m

        if verbose: print("[{}] name: {} => values: \n{}\n ... mode: {}".format(i+1, col, mv, mn))
        adict[col] = mv
    # [2] name: patient_state => values: 
    # [('CA', 18128), ('TX', 17501), ('FL', 16775), ('NY', 11341), (nan, 7817), ('NJ', 6739), ('PA', 6739), ('GA', 5308), ('MD', 4947), ('NC', 3427)]
    #  ... mode: CA
    return adict

def det_cardinality2(df, **kargs): 
    th_card = kargs.get('th_card', 10) # if n(uniq values) >= this number => high card
    assert th_card > 0
    adict = det_cardinality(df, **kargs)

    header = ['high_card', 'low_card', 'numeric', 'ordinal', 'text', 'long_text', ]
    d = {h:[] for h in header}

    # 
    for col, tval in adict.items(): 
        if tval: 
             d['high_card'].append(col)
        else: 
            d['low_card'].append(col)

    return d

def stratify(df, col='test_result_loinc_code', ascending=False): 
    import operator
    assert col in df.columns
    
    ds = dict(df[col].value_counts())
    ds = sorted(ds.items(), key=operator.itemgetter(1), reverse=not ascending)
    
    return ds

def balance_by_downsampling(X, y, method='median', majority_max=3): 
    """
    Params
    ------
        majority_max: The sample size of the ajority class can at most be this multiple of the minority-class sample size
                      e.g. suppose that majority_class = 3, then ... 

    """
    import pandas as pd
    import collections
    
    nf = X.shape[1]
    labels = np.unique(y)
    label_id = nf+1
    lcnt = collections.Counter(y) # label counts

    lastn = 1
    Nmin = lcnt.most_common()[:-lastn-1:-1][0][1]
    Nmax = lcnt.most_common(1)[0][1]
    
    if len(y.shape) == 1: # 1-D array 
        assert X.shape[0] == len(y)
        X = np.hstack([X, y[:, None]])
    else: 
        assert X.shape[0] == y.shape[0]
        X = np.hstack([X, y])
    
    ###########
    ts = DataFrame(X)
    if method.startswith('med'): # median
        Ncut = int(np.median([c for l, c in lcnt.items()]))

    elif method.startswith('multi'):  # multiple
        Ncut = Nmin * majority_max

    ###########
    tx = []
    for label in labels: 
        tsl = ts[ts[label_id]==label]
        if tsl.shape[0] > Ncut:
            tx.append(tsl.sample(n=Ncut))
        else: 
            tx.append(tsl) 

    if len(tx) > 0: 
        ts = pd.concat(tx, ignore_index=True) 

    # separate ts into (X, y)
    X = ts.iloc[:,:nf]
    y = ts.iloc[label_id]

    return (X, y)

def balance_data_incr(df, df_extern, n_samples=-1, col='test_result_loinc_code', labels=[], verbose=1, **kargs):
    """
    Similar to balance_data() but expect df2 to be huge (to a degree that may not fit into main memory)
    """
    import loinc as ul

    n_baseline = n_samples
    if n_baseline < 0: 
        ds = stratify(df, col=col)
        # use the median sample size as the standard sample size
        n_baseline = int(np.median([e[1] for e in ds]))
    print("(balance_data_incr) n_baseline={}".format(n_baseline))

    # target labels from the given 'col'
    if len(labels) == 0: 
        labels = set(df[col])

        if df_extern is not None or not df_extern.empty:
            labels_external = set(df_extern[col])
            print("(balance_data_incr) Found n={} unique codes from source | nc={} unique codes from external".format(
                len(labels), len(labels_external)))

            delta_l = list(set(labels_external)-set(labels))
            print("... found {}=?=0 extra codes from the df_extern:\n{}\n".format(len(delta_l), delta_l[:20]))
            labels_common = list(set(labels_external).intersection(labels))
            print("... found {} common codes from the df_extern:\n{}\n".format(len(labels_common), labels_common[:20]))

    N0 = df.shape[0]
    assert df.shape[1] == df_extern.shape[1], "Inconsistent feature dimension: nf={} <> nf2={}".format(df.shape[1], df_extern.shape[1])

    ###############################################
    delta = {}
    dfx = []
    n_hit = n_miss = 0. # n_extra
    hit, missed = set([]), set([])
    for k, dfi in df.groupby([col, ]): 
        if not k in labels: continue # don't add data if not in the target label set
        if k in ul.LoincTSet.non_codes: continue  # don't add data if the classe is either 'unknown' or 'others'

        ni = dfi.shape[0]
        dfj = df_extern.loc[df_extern[col]==k]
        nj = dfj.shape[0]

        if nj == 0: # cannot find matching rows in df_extern
            # print(f"(balance_training_data) Could not find code={r} in external data ... !!!")
            missed.add(k)
            # if verbose and len(missed)%50==0: 
            #     print("(balance_data_incr) Could not find cases in external for code={} (n={})... #".format(k, len(missed)))
        else: 
            hit.add(k)

            if n_baseline > ni: 
                delta = n_baseline-ni
                dfx.append( dfj.sample(n=min(delta, nj)) ) 

                if verbose and len(hit) % 25 == 0: 
                    print("(balance_data_incr) Added n={} cases to code={}".format(dfx[-1].shape[0], k))
    
    if verbose: 
        print("(balance_data_incr) Added n={} cases in total | n_miss:{} ... #".format(len(hit), len(missed))) # [debug] somehow the values got cast to float
        print("... missed (n={}):\n{}\n".format(len(missed), list(missed)[:20]))
        print("... hit (n={}):\n{}\n".format(len(hit), list(hit)[:20]))

    df_incr = DataFrame(columns=df.columns)
    if len(dfx) > 0: 
        df_incr = pd.concat(dfx, ignore_index=True)
        # df = pd.concat([df, df_extra], ignore_index=True)

    if verbose: 
        print("(balance_data) N_extra: {}".format(df_incr.shape[0])) 

    return df_incr, list(hit), list(missed)

def balance_data(df, df2=None, n_samples=-1, col='test_result_loinc_code', labels=[], verbose=1): 
    """
    Make class-balanced training data by balancing sample sizes.

    Input
    -----
    df: Input dataframe whose columns contain both features/variables and class labels

    """
    if n_samples < 0: 
        ds = stratify(df, col=col)
        # use the median sample size as the standard sample size
        n_samples = int(np.median([e[1] for e in ds]))
    print("(balance_data) n_samples={}".format(n_samples))

    # target labels from the given 'col'
    if len(labels) == 0: 
        labels = set(df[col])

        if df2 is not None or not df2.empty:
            labels_external = set(df2[col])
            print("(balance_data) Found n={} unique codes from source | nc={} unique codes from external".format(
                len(labels), len(labels_external)))
        
    N0 = df.shape[0]

    if verbose: 
        labels = np.random.choice(df[col].unique(), 30)
        print("(balance_data) example labels:\n{}\n".format(list(labels)))
        if df2 is not None or not df2.empty: 
            labels2 = np.random.choice(df2[col].unique(), 30)
            print("(balance_data) example labels from supplement data:\n{}\n".format(list(labels2)))
    
    dfx = []
    if df2 is None or df2.empty: # Case 1: Balance within the original training data (df) without additional data (df2)
    
        for r, df in df.groupby([col, ]): 
            n = df.shape[0]
            if n > n_samples: 
                dfx.append( df.sample(n=n_samples) )
            else: 
                dfx.append( df )

        df = pd.concat(dfx, ignore_index=True)
    else: ## case 2: Balance with additional training data (df2)
        assert df.shape[1] == df2.shape[1], "Inconsistent feature dimension: nf={} <> nf2={}".format(df.shape[1], df2.shape[1])

        delta = {}
        dfx = []
        n_hit = n_miss = 0. # n_extra
        hit, missed = [], []
        for r, dfi in df.groupby([col, ]): 
            if not r in labels: continue

            n = dfi.shape[0]

            dfc = df2.loc[df2[col]==r]
            nc = dfc.shape[0]
            if nc == 0: 
                n_miss += 1
                # print(f"(balance_training_data) Could not find code={r} in external data ... !!!")
                missed.append(r)
            else: 
                hit.append(r)
            
            if n >= n_samples: 
                pass # no need to add data 
                # delta[r] = 0
            else: 
                delta = n_samples-n

                if nc > 0: 
                    # delta[r] = n_samples-n
                    dfx.append( dfc.sample(n=min(delta, nc)) )
                    n_hit += 1 

                    if verbose and n_hit < 20: 
                        print("(balance_data) Added n={} cases to code={}".format(dfx[-1].shape[0], r))
                else: 
                    print(f"... Could not find cases in external for code={r} ... #")
        
        print("(balance_data) Added n={} cases in total | n_miss:{} ... #".format(n_hit, n_miss)) # [debug] somehow the values got cast to float
        print(f"... missed:\n{missed}\n")
        print(f"... hit:\n{hit}\n")

        if len(dfx) > 0: 
            df_extra = pd.concat(dfx, ignore_index=True)
            df = pd.concat([df, df_extra], ignore_index=True)

        if verbose: 
            print("(balance_data) N: {} => {}".format(N0, df.shape[0])) 

    return df


########################################

def t_performance(**kargs): 
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    print("> before y:\n{}\n".format(y))
    y = encode_labels(y, pos_label=1)

    print("> after  y:\n{}\n".format(y))

    scores = eval_performance(X, y, model=None, cv=5, random_state=53)
    print("> average: {}, std: {}".format(np.mean(scores), np.std(scores)))

    return scores

def plot_barh(perf, metric='fmax', **kargs): 
    """

    Reference
    ---------
    1. bar annotation  
        + http://robertmitchellv.com/blog-bar-chart-annotations-pandas-mpl.html

        + Annotate bars with values on Pandas bar plots
            https://stackoverflow.com/questions/25447700/annotate-bars-with-values-on-pandas-bar-plots

            <idiom> 
            for p in ax.patches:
                ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

        + adding value labels on a matplotlib bar chart
            https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart

    2. colors:  
        + https://stackoverflow.com/questions/11927715/how-to-give-a-pandas-matplotlib-bar-graph-custom-colors

        + color map
          https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib

        + creating custom colormap: 
            https://matplotlib.org/examples/pylab_examples/custom_cmap.html

    3. fonts: 
        http://jonathansoma.com/lede/data-studio/matplotlib/changing-fonts-in-matplotlib/
    """
    from utils_plot import truncate_colormap 
    from itertools import cycle, islice
    import matplotlib.colors as colors
    from matplotlib import cm

    # perf: PerformanceMetrics object

    plt.clf()

    # performance metric vs methods
    codes = kargs.get('codes', [])  # [] to plot all codes

    # perf_methods
    perf_codes = perf.table.loc[metric] if not methods else perf.table.loc[metric][methods]  # a Series
    if kargs.get('sort', True): 
        perf_methods = perf_methods.sort_values(ascending=kargs.get('ascending', False))

    # default configuration for plots
    matplotlib.rcParams['figure.figsize'] = (10, 20)
    matplotlib.rcParams['font.family'] = "sans-serif"

    # canonicalize the method/algorithm name [todo]

    # coloring 
    # cmap values: {'gnuplot2', 'jet', }  # ... see demo/coloarmaps_refeference.py
    ####################################################################
    # >>> change coloring options here
    option = 2
    if option == 1: 
        cmap = truncate_colormap(plt.get_cmap('jet'), 0.2, 0.8) # params: minval=0.0, maxval=1.0, n=100)

        # fontsize: Font size for xticks and yticks
        ax = perf_methods.plot(kind=kargs.get('kind', "barh"), colormap=plt.get_cmap('gnuplot2'), fontsize=12)  # colormap=cmap 
    elif option == 2: 
        # Make a list by cycling through the desired colors to match the length of your data
        
        # methods_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(perf_methods)))
        if 'color_indices' in kargs: 
            method_colors = kargs['color_indices']
        else: 
            n_models = len(perf_methods)
            method_colors = cm.gnuplot2(np.linspace(.2,.8, n_models+1))  # this is a 2D array

        ax = perf_methods.plot(kind=kargs.get('kind', "barh"), color=method_colors, fontsize=12)  # colormap=cmap 
    elif option == 3:  # fixed color 
        # my_colors = ['g', 'b']*5 # <-- this concatenates the list to itself 5 times.
        # my_colors = [(0.5,0.4,0.5), (0.75, 0.75, 0.25)]*5 # <-- make two custom RGBs and repeat/alternate them over all the bar elements.
        method_colors = [(x/10.0, x/20.0, 0.75) for x in range(len(perf_methods))] # <-- Quick gradient example along the Red/Green dimensions.
        ax = perf_methods.plot(kind=kargs.get('kind', "barh"), color=method_colors, fontsize=12)  # colormap=cmap 
    ####################################################################

    # title 
    x_label = '%s scores' % metric.upper()
    y_label = "LOINC in descending order of performance"
    ax.set_xlabel(x_label, fontname='Arial', fontsize=12) # color='coral'
    ax.set_ylabel(y_label, fontname="Arial", fontsize=12)
    
    ## annotate the bars
    # Make some labels.
    labels = scores = perf_methods.values
    min_abs, max_abs = kargs.get('min', 0.0), kargs.get('max', 1.0)
    min_val, max_val = np.min(scores), np.max(scores)
    epsilon = 0.1 
    min_x, max_x = max(min_abs, min_val-epsilon), min(max_abs, max_val+epsilon)

    # [todo] set upperbound according to the max score
    ax.set_xlim(min_x, max_x)

    # annotation option
    option = 1
    if option == 1: 
        for i, rect in enumerate(ax.patches):
            x_val = rect.get_x()
            y_val = rect.get_y()

            # if i % 2 == 0: 
            #     print('... x, y = %f, %f  | width: %f, height: %f' % (x_val, y_val, rect.get_width(), rect.get_height()))   # [log] x = 0 all the way? why? rect.get_height = 0.5 all the way
            # get_width pulls left or right; get_y pushes up or down
            # memo: get_width() in units of x ticks
            assert rect.get_width()-scores[i] < 1e-3

            width = rect.get_width()
            # lx_offset = width + width/100. 

            label = "{:.3f}".format(scores[i])
            ax.text(rect.get_width()+0.008, rect.get_y()+0.08, \
                    label, fontsize=8, color='dimgrey')
    elif option == 2: 
        # For each bar: Place a label
        for i, rect in enumerate(ax.patches):
            # Get X and Y placement of label from rect.
            y_value = rect.get_y()  # method  ... rect.get_height()
            x_value = rect.get_x()  # score

            # Number of points between bar and label. Change to your liking.
            space = 5

            # Use Y value as label and format number with one decimal place
            label = "{:.2f}".format(x_value) # "{:.1f}".format(y_value)

            # Create annotation
            plt.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va='center')                      # Vertically align label differently for
                #                             # positive and negative values.

    # invert for largest on top 
    # ax.invert_yaxis()  # 

    plt.yticks(fontsize=8)
    plt.title("Performance Comparison In %s" % metric.upper(), fontname='Arial', fontsize=15) # fontname='Comic Sans MS'

    n_methods = perf.n_methods()
    
    # need to distinguish the data set on which algorithms operate
    domain = perf.query('domain')
    if domain == 'null': 
        assert 'domain' in kargs
        domain = kargs['domain']

    file_name = kargs.get('file_name', '{metric}_comparison-N{size}-D{domain}'.format(metric=metric, size=n_methods, domain=domain))  # no need to specify file type
    if 'index' in kargs: 
        file_name = '{prefix}-{index}'.format(prefix=file_name, index=kargs['index'])
    saveFig(plt, PerformanceMetrics.plot_path(name=file_name), dpi=300)  # basedir=PerformanceMetrics.plot_path

    return

def t_io(**kargs):

    cohort = 'hepatitis-c'
    # MTRT dataframe / training data
    df = load_data(cohort=cohort, verbose=1)  # the original training data with relevant MTRT attributes

    return 

def t_analyze(**kargs):
    categories = ['easy', 'hard', 'low']  # low: low sample size

    ccmap = label_by_performance(cohort='hepatitis-c', categories=categories)

    # a dictionary of LOINC table dataframes indexed by analysis categories (e.g. easy, hard)
    D = analyze_by_values(ccmap, verbose=1) #  

    return 

def t_stratify(**kargs): 
    import sys
    from transformer import to_age
    # from loinc import canonicalize
    import loinc as lc

    cohort = 'hepatitis-c'
    token_default = token_missing = 'unknown'
    token_other = 'other'
    col_target = 'test_result_loinc_code'
    tBigData = True
    tSave = True

    df_perf = load_performance(input_dir='result', cohort=cohort)

    # ... list of 2-tuples: [(code<i>, count<i>) ... ]
    max_size = np.max(df_perf['n_pos'].values)
    print("(t_stratify) control class sample size to be: {}".format(max_size))

    

    # load training data 
    # cohort = 'loinc-hepatitis-c'

    # ts0 = load_data(cohort=cohort)  
    # ... this loads curated dataset 

    # load source data
    ts0 = load_data(input_file='andromeda-pond-hepatitis-c.csv', warn_bad_lines=False)
    ts0 = ts0.drop_duplicates(keep='last')  # drop duplicates 
    ts0 = lc.canonicalize(ts0, col_target=col_target, token_missing=token_default)
    N0 = ts0.shape[0]

    print("(t_stratify) dim(ts0): {}".format(ts0.shape))  
    loinc_set = codes0 = ts0[col_target].unique()
    codes0_subset = np.random.choice(codes0, 30)
    print("(t_stratify) example codes (source):\n{}\n".format(list(codes0_subset)))

    # load source data
    ts_ctrl = load_data(input_file='andromeda_pond-10p.csv', warn_bad_lines=False)
    ts_ctrl = ts_ctrl.drop_duplicates(keep='last')  # drop duplicates
    ts_ctrl = lc.canonicalize(ts_ctrl, col_target=col_target, token_missing=token_default, target_labels=loinc_set)
    print("> dim(ts_ctrl): {}".format(ts_ctrl.shape))  

    Nctrl = ts_ctrl.shape[0]
    assert Nctrl > N0, f"Control data is too small | n({cohort})={N0} > n(ctrl)={Nctrl}"
    ts_ctrl = ts_ctrl.sample(n=N0, replace=False)
    ts0 = pd.concat([ts0, ts_ctrl], ignore_index=True)

    ########################
    # ts0: positive + extra control
    assert np.sum(ts0[col_target].isnull()) == 0

    ds = stratify(ts0, col='test_result_loinc_code', ascending=False)
    print("(balance_classes) data size distribution:\n{}\n".format(ds[:25]))
    ds = [(code, size) for code, size in ds if not code in (token_default, token_other,)]

    max_size = ds[0][1]
    codes_low_sz = set([code for code, size in ds if size < max_size])
    print("(balance_classes) We have n={} with low sample size (< {})".format(len(codes_low_sz), max_size))

    # sys.exit(0)
    print("-" * 50 + "\n")

    # Case 1: no extra data
    # ts = balance_data(df=ts0, df2=None, n_samples=100, col=col_target)
    # print("(t_stratify) dim(ts): {}".format(ts.shape))  
    # for r, s in ts[col_target].value_counts().items(): 
    #     print(f"[{r}] {s}")
    if not tBigData: 
        ts2 = load_data(input_file='andromeda-pond-hepatitis-c-loinc.csv')
        ts2 = ts2.drop_duplicates(keep='last')  # drop duplicates 
        ts2 = lc.canonicalize(ts2, col_target=col_target, token_missing=token_default, columns=ts0.columns, target_labels=loinc_set) # noisy_values/[]
        print("(t_stratify) dim(ts2): {}".format(ts2.shape))  
        codes2 = ts2[col_target].unique()
        codes2_subset = np.random.choice(codes2, 30)
        print("(t_stratify) example codes (external):\n{}\n".format(list(codes2_subset)))

        codes_delta = list(set(codes2)-set(codes0))
        print("... found {}=?=0 extra codes from the ts2:\n{}\n".format(len(codes_delta), codes_delta))
        codes_common = list(set(codes2).intersection(codes0))
        print("... found {} common codes from the ts2:\n{}\n".format(len(codes_common), codes_common))
        
        # ts2 = to_age(ts2)
        # ts2.fillna(value='unknown', inplace=True)

        # print("(t_stratify) dim(ts2): {}".format(ts2.shape))

        ts = balance_data(df=ts0, df2=ts2, n_samples=max_size, col=col_target)
        print("(t_stratify) dim(ts): {}".format(ts.shape))  
        # for r, s in ts[col_target].value_counts().items(): 
        #     print(f"[{r}] {s}")
    else: 
        # read the large csv file with specified chunksize 
        # max_size = 5000
        ts = ts0
        codes = set(loinc_set)

        codes_hit = set([])
        for i, tsi in enumerate(load_data_incr(input_file='andromeda-pond-hepatitis-c-loinc.csv', chunksize=1000000, warn_bad_lines=False)): 
            N0 = ts.shape[0]
            tsi = tsi.drop_duplicates(keep='last')  # drop duplicates 
            tsi = lc.canonicalize(tsi, col_target=col_target, token_missing=token_default, target_labels=codes)
            print("[{}] Processing chunk #{} | n(ts): {}, n(tsi): {} ...".format(i, i+1, N0, tsi.shape[0]))

            ts_incr, hit, missed = balance_data_incr(df=ts, df_extern=tsi, n_samples=max_size, col=col_target)
            if not ts_incr.empty: ts = pd.concat([ts, ts_incr])

            # analysis 
            N = ts.shape[0]
            codes_hit.union(hit) # still has this many codes without a match 

            print(f"[{i}] size: {N0} -> {N}")
            ds = stratify(ts, col='test_result_loinc_code', ascending=False)
            ds = [(code, size) for code, size in ds if size < max_size]
            print("[{}] size(codes) < {} (n={}): \n{}\n".format(i, max_size, len(ds), ds[:50]))

    codes_missed = codes_low_sz - codes_hit  
    print("(t_stratify) At last, we could not find a match for n={} codes among nl={} low-sample-size labels:\n{}\n".format(
        len(codes_missed), len(codes_low_sz), codes_missed))  

    if tSave: 
        ts = ts.drop_duplicates(keep='last')  # drop duplicates 
        output_file = f"andromeda-pond-{cohort}-balanced.csv"
        save_data(ts, output_file=output_file)

    return

def t_loinc(**kargs):
    from transformer import dehyphenate, trim_tail, replace_values

    df = load_src_data(cohort='hepatitis-c') 

    print("> Source df dim: {}".format(df.shape))

    col_target = "test_result_loinc_code"
    token_default = 'unknown'
    
    noisy_values = [] # ['request', 'no loinc needed', '.', ';', 'coumt', 'unloinc']

    ##############################
    df[col_target].fillna(value=token_default, inplace=True)
    dehyphenate(df, col=col_target)
    # replace_values(df, values=noisy_values, new_value=token_default, col=col_target)
    trim_tail(df, col=col_target, delimit=['.', ';', ])
    df[col_target].replace('', token_default, inplace=True) 

    print("> Found n={} rows where loinc: empty string".format(df[df[col_target] == ''].shape[0]))

    ##############################

    for code in df[col_target].values:
        try: 
            int(code)
        except: 
            if code in [token_default, ]: 
                pass 
            else:
                noisy_values.append(code)

    noisy_values = list(set(noisy_values))
    print("> Found n={} noisy codes:\n{}\n".format(len(noisy_values), noisy_values))

    # empty strings 


    return

def test(**kargs):
    # input_dir = os.getcwd()
    # input_file = "andromeda_dataset.csv"
    # input_path = os.path.join(input_dir, input_file)
    # df = pd.read_csv(input_path, sep=',', header=0)
 
    # msg = ""
    # msg += "> sample sizes: {}\n".format(df.shape[0])
    # msg += "> n(features):  {}\n".format(df.shape[1])
    
    # print(msg)

    # t_performance(**kargs)

    ### I/O operations
    # t_io(**kargs)

    ### Analysis 
    # t_analyze(**kargs)

    ### Stratigy training data 
    t_stratify()   # ... ok 


    ### LOINC codes analysis
    # t_loinc(**kargs)


    return

if __name__ == "__main__":
    test()