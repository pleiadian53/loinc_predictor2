import pandas as pd 
import numpy as np
from pandas import DataFrame, Series
import os, re, collections

from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor    # Import decision tree classifier model
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from tabulate import tabulate
import common
import config
import data_processor as dproc

from utils_sys import highlight
from utils_plot import saveFig # contains "matplotlib.use('Agg')" which needs to be called before pyplot 
from matplotlib import pyplot as plt

from loinc import LoincTable, LoincTSet, FeatureSet
from loinc_mtrt import LoincMTRT
from loinc import load_loinc_table
from loinc_mtrt import load_loinc_to_mtrt

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

def col_values_by_codes(codes, df=None, cols=['test_result_name', 'test_order_name'], **kargs):
    cohort = kargs.get('cohort', 'hepatitis-c')
    verbose = kargs.get('verbose', 1)
    mode = kargs.get('mode', 'raw')   # {'raw', 'unique', 'sample'}
    n_samples = kargs.get('n', 10)

    col_target = 'test_result_loinc_code'
    if df is None: 
        processed = kargs.get('processed', True)
        canonicalized = kargs.get('canonicalized', True)
        df = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=canonicalized, processed=processed)

    df = df.loc[df[col_target].isin(codes)]
    print("(data) Given codes (n={}), we get N={} rows from the training data.".format(len(codes), df.shape[0]))
    
    adict = {}
    for col in cols: 
        if mode.startswith('r'):  # raw, as it is
            adict[col] = df[col].values
        elif mode.startswith('u'):
            adict[col] = df[col].unique()
        else: 
            adict[col] = df.sample(n=n_samples, random_state=kargs.get('random_state', 53))[col].values
    return adict
       
def col_values2(df, cols=[], n=10, mode='sampling', random_state=1, keep='last'):
    # similar to col_values but returns dataframe instead of Series or np.array
    if mode.startswith( ('s', 'r') ):  # sampling, random
        n = min(n, df.shape[0])
        return df.sample(n=n, random_state=random_state)[cols]
    elif mode.startswith('u'):  # unique values
        return df[cols].drop_duplicates(keep=keep)
    return df.sample(n=n, random_state=random_state)[cols]

def col_values(df, col='age', n=10, mode='sampling', random_state=1):
    if mode.startswith( ('s', 'r') ):  # sampling, random
        n = min(n, df.shape[0])
        return df.sample(n=n, random_state=random_state)[col].values
    elif mode.startswith('u'):  # unique values
        return df[col].unique()
    return df.sample(n=n, random_state=random_state)[col].values

def sample_col_values(df, col='age', n=10, random_state=1):
    return col_values(df, col=col, n=n, random_state=random_state, mode='sampling')

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

def show_dict(adict, topn=-1, by='', header=[], n_samples=-1, ascending=False, print_=False): 
    # print(adict)
    
    # convert to two-column dataframe format 
    if not header: 
        header = ['key', 'value']
    else: 
        assert len(header) == 2
    D = {h:[] for h in header}
    for k, v in adict.items(): 
        D[header[0]].append(k)
        D[header[1]].append(v)
    
    df = DataFrame(D, columns=header)
    msg = ''
    if topn > 0: 
        assert by in df.columns
        df = df.sort_values([by, ], ascending=ascending)
        msg = tabulate(df[:topn], headers='keys', tablefmt='psql')
    else: 
        if n_samples < 0: 
            msg = tabulate(df, headers='keys', tablefmt='psql')
        else: 
            n = min(df.shape[0], n_samples)
            msg = tabulate(df.sample(n=n), headers='keys', tablefmt='psql')
    if print_: print(msg)
    return msg

# Classifier Utilties
#######################################################

def transform_and_encode(df, fill_missing=True, token_default='unknown', 
        drop_high_missing=False, pth_null=0.9, verbose=1):
    """
    Transform the non-matching variables (i.e. all variables NOT used to match
    with the LOINC decriptions such as meta_sender_name, test_order_code, test_result_code). 

    Matching variables are the T-attributes that carry text values (e.g. test_order_name, test_result_name)
    Non-matching variables are typically not text-valued columns

    Input
    -----
    df: source training data 

    """
    from transformer import encode_vars 

    # matchmaker features 
    cat_cols = FeatureSet.cat_cols
    cont_cols = FeatureSet.cont_cols
    target_cols = FeatureSet.target_cols
    high_card_cols = FeatureSet.high_card_cols

    # --- transform variables
    FeatureSet.to_age(df)
    values = col_values(df, col='age', n=10)
    print("[transform] age: {}".format(values))

    # -- Categorize variables
    regular_vars, target_vars, derived_vars, meta_vars = FeatureSet.categorize_features(df)
    # ...note 
    #    regular_vars: non-matching columns (e.g. meta_sender_name, test_order_code, test_result_code, ...)

    # V = cont_cols + cat_cols  # + derived_cols (e.g. count)
    # L = target_cols
    dfX = df[regular_vars]
    dfY = df[target_vars]

    # optinal variables
    dfD = df[derived_vars] if len(derived_vars) > 0 else DataFrame()
    dfZ = df[meta_vars] if len(meta_vars) > 0 else DataFrame()

    if fill_missing:  
        # dfM.fillna(value=token_default, inplace=True)
        # ... don't fill missing values for dfM here!
        dfX[cont_cols].fillna(value=0, inplace=True)
        dfX[cat_cols].fillna(value=token_default, inplace=True)

        dfY.fillna(value=token_default, inplace=True)

    if drop_high_missing: 
        # drop columns/vars with too many missing values 
        N = dfX.shape[0]
        n_thresh = int(N * pth_null)
        nf0 = nf = dfX.shape[1]
        fset0 = set(dfX.columns.values)

        dfX = dfX[dfX.columns[dfX.isnull().mean() < pth_null]]
        fset = set(dfX.columns.values)
        nf = dfX.shape[1]
        print("[transform] Dropped n={} features:\n{}\n".format(nf-nf0, fset0-fset))

    dim0 = dfX.shape
    dfX, encoder = encode_vars(dfX, fset=cat_cols, high_card_cols=high_card_cols)
    if verbose: 
        print("[encode] dim(dfX-): {}, dim(dfX+): {}".format(dim0, dfX.shape))
        # [log] ... [encode] dim(dfX-): (64979, 13), dim(dfX+): (64979, 97)

    return (dfX, dfY, dfD, dfZ, encoder)

def feature_transform(df, **kargs):
    # from loinc import LoincTSet
    tDropHighMissing = kargs.get('drop_high_missing', False)
    pth_null = kargs.get("pth_null", 0.9)  # threshold of null-value proportion to declare a "high missing rate"
    verbose = kargs.get('verbose', 1)
    token_default = kargs.get("token_default", LoincTSet.token_default)

    N0, Nv0 = df.shape
    dfX, dfY, dfD, dfZ, encoder = \
         transform_and_encode(df, fill_missing=True, token_default=token_default, 
                drop_high_missing=tDropHighMissing, pth_null=pth_null)
    # ... transform and encode only deals with X i.e. regular variables (i.e. non-matching variables)
    
    df = pd.concat([dfX, dfY, dfD, dfZ], axis=1) # matching, regular, target, derived, meta  
    
    assert df.shape[0] == N0
    if verbose: highlight("[transform] dim of vars: {} -> {}".format(Nv0, df.shape[1]))
    return df

def run_model_selection(X, y, model, p_grid={}, n_runs=30, scoring='roc_auc', output_path='', output_file='', **kargs): 
    from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
    
    create_dir = kargs.get("create_dir", True)
    meta = kargs.get("meta", "")
    plot_ = kargs.get("plot", True)
    ext = kargs.get("ext", "tif"),
    index = kargs.get("index", 0)
    save_plot = kargs.get("save", False)  # save
    verbose = kargs.get("verbose", 1)
    dpi = kargs.get('dpi', 300)

    # Arrays to store scores
    non_nested_scores = np.zeros(n_runs)
    nested_scores = np.zeros(n_runs)

    # Loop for each trial
    icv_num = 5
    ocv_num = 5
    params = {}  # trial to best params
    for i in range(n_runs): 

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
        params[i] = clf.best_params_

    score_difference = non_nested_scores - nested_scores

    print("[model_selection] Average difference of {:6f} with std. dev. of {:6f}."
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
        difference_plot = plt.bar(range(n_runs), score_difference)
        plt.xlabel("Individual Trial #")
        plt.legend([difference_plot],
                   ["Non-Nested CV - Nested CV Score"],
                   bbox_to_anchor=(0, 1, .8, 0))
        plt.ylabel("score difference", fontsize="14")

        if save_plot: 
            from utils_plot import saveFig
            if not output_path: output_path = os.path.join(os.getcwd(), 'analysis')
            if not os.path.exists(output_path) and create_dir:
                print('(run_model_selection) Creating analysis directory:\n%s\n' % output_path)
                os.mkdir(output_path) 

            if output_file is None: 
                classifier = 'DT'
                name = 'ModelSelect-{}'.format(classifier) 
                if meta: 
                    output_file = '{prefix}.E-{suffix}.{ext}'.format(prefix=name, suffix=meta, ext=ext)
                else:
                    output_file = '{prefix}.{ext}'.format(prefix=name, ext=ext)

            output_path = os.path.join(output_path, output_file)  # example path: System.analysisPath

            if verbose: print('(run_model_selection) Saving model-selection-comparison plot at: {path}'.format(path=output_path))
            saveFig(plt, output_path, dpi=dpi, verbose=verbose) 
        else: 
            plt.show()
        
    best_index = np.argmax(nested_scores)
    best_params = params[best_index]
    return best_params, params, nested_scores

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

    if len(y.shape) == 1: 
        y = y.reshape((y.shape[0], 1))

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


### I/O Utilities ###

def save_generic(df, cohort='', dtype='ts', output_file='', sep=',', **kargs):
    return dproc.save_generic(df, cohort=cohort, dtype=dtype, output_file=output_file, sep=sep, **kargs)
def load_generic(cohort='', dtype='ts', input_file='', sep=',', **kargs):
    return dproc.load_generic(cohort=cohort, dtype=dtype, input_file=input_file, sep=sep, **kargs)  

def save_data(df, cohort='', output_file='', sep=',', **kargs): 
    return dproc.save_data(df, cohort=cohort, output_file=output_file, sep=sep, **kargs)
def load_data(cohort='', input_file='', sep=',', **kargs):
    return dproc.load_data(cohort=cohort, input_file=input_file, sep=sep, **kargs)

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

def load_src_data(**kargs): 
    return dproc.load_src_data(**kargs)

def load_performance(input_dir='result', input_file='', **kargs):
    cohort = kargs.get('cohort', 'hepatitis-c')
    sep = kargs.get('sep', '|')

    if not input_dir: input_dir = os.path.join(os.getcwd(), 'result') # os.path.join(os.getcwd(), 'result')
    if not input_file: input_file = f"performance-{cohort}.csv" 
    input_path = os.path.join(input_dir, input_file)
    assert os.path.exists(input_path), "Invalid path: {}".format(input_path)

    df = pd.read_csv(input_path, sep=sep, header=0, index_col=None, error_bad_lines=False)
    print("> dim(performance matrix): {}".format(df.shape)) 

    return df
    
def save_performance(df, output_dir='result', output_file='', **kargs): 
    cohort = kargs.get('cohort', 'hepatitis-c')
    sep = kargs.get('sep', '|')
    verbose = kargs.get('verbose', 1)

    output_dir = kargs.get('output_dir', os.path.join(os.getcwd(), output_dir)) 
    if not output_file: output_file = f"performance-{cohort}.csv" 
    output_path = os.path.join(output_dir, output_file)
    df.to_csv(output_path, sep=sep, index=False, header=True)

    if verbose: 
        print('(save) Saving performance dataframe to:\n{}\n ... #'.format(output_path))
        for code, score in zip(df['code'], df['mean']):
            print(f"[{code}] -> {score}")
    return

def label_by_types(df=None, cohort='hepatitis-c', categories=[], processed=False, transformed_vars_only=True): 
    """
    Types of LOINC codes to predict according to the absence and presence of loinc codes and MTRT.

    Memo
    ----
    0: absent, 1: present

    LOINC    MTRT    type 
      1      1        I 
      0      1        II
      1.     0        III 
      0.     0.       IV

    # heptatis C 
    ... Type(1): N=67079
    ... Type(2): N=0
    ... Type(3): N=4145
    ... Type(4): N=0


    """
    from loinc import LoincTSet
    if df is None: 
        df = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=True, processed=processed)
    
    col_target = LoincTSet.col_target  # loinc codes as class labels
    col_tag = LoincTSet.col_tag    # MTRTs as tags

    adict = {}
    adict[1] = df.loc[~df[col_target].isnull() & ~df[col_tag].isnull()].index.values
    adict[2] = df.loc[df[col_target].isnull() & ~df[col_tag].isnull()].index.values
    adict[3] = df.loc[~df[col_target].isnull() & df[col_tag].isnull()].index.values
    adict[4] = df.loc[df[col_target].isnull() & df[col_tag].isnull()].index.values

    if not transformed_vars_only: 
        df['prediction_level'] = 1
        for level in adict.keys(): 
            df.loc[adict[level], 'prediction_level'] = level

    return (df, adict)  # adict: holds entries to these different types of samples; as a return value only for convenience

def label_by_performance(cohort='hepatitis-c', th_low=0.50, th_high=0.90, categories=[], verify=True, verbose=1): 
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

def det_range2(X, y, fset, **kargs):
    def the_other_label(y, given): 
        ulabels = list(np.unique(y))
        ulabels.remove(given)
        return ulabels

    # output 
    res = {}   # used to collect the statistics for later use
    uniq_labels = np.unique(y)

    # special case: binary
    isSimpleBinary = True
    lmap = {}
    if len(uniq_labels) == 2: 
        lmap = {1: '+', 0: '-', -1: '-'}
        isSimpleBinary = True

    vmin, vmax = np.min(X), np.max(X)
    print("[analysis] min(X): {}, max(X): {}".format(vmin, vmax))
    # print("... dim(X): {}, dim(y): {}".format(X.shape, y.shape))
    res['Xmin'], res['Xmax'] = vmin, vmax

    for label in uniq_labels: 
        polarity = lmap.get(label, '?')
        print("... column-wise mean ({}):\n{}\n".format(polarity, list(zip(fset, np.mean(X[y==label, :], axis=0)))))

    print("... column-wise mean overall:\n{}\n".format(list(zip(fset, np.mean(X, axis=0)))))

    # row_mean = np.mean(X, axis=1)
    # print("... n(row-wise 0): {}".format(np.sum(row_mean == 0)))

    # look into "trival vectors" such as zero vectors
    if isSimpleBinary: 
        pos_label = kargs.get('pos_label', 1)
        neg_label = the_other_label(uniq_labels, given=pos_label)[0]

        # assume that the positive is encoded as 1
        X_pos = X[y==pos_label, :]
        X_neg = X[y==neg_label, :]
    
        rows_zeros = np.where(~X.any(axis=1))[0]
        print("... found n={} zero vectors overall (which cannot be normalized; troubles in computing correlation)".format(len(rows_zeros)))
        rows_pos_zeros = np.where(~X_pos.any(axis=1))[0]
        rows_neg_zeros = np.where(~X_neg.any(axis=1))[0]
        print("... N(pos): {}, n(pos, 0): {}, ratio: {}".format(X_pos.shape[0], len(rows_pos_zeros), len(rows_pos_zeros)/(X_pos.shape[0]+0.0)))
        print("... N(neg): {}, n(neg, 0): {}, ratio: {}".format(X_neg.shape[0], len(rows_neg_zeros), len(rows_neg_zeros)/(X_neg.shape[0]+0.0)))

    return res

def det_range(df, **kargs): 
    def the_other_label(y, given): 
        ulabels = list(np.unique(y))
        ulabels.remove(given)
        return ulabels

    from data_processor import toXY
    cols_y = kargs.get("cols_y", [])
    cols_x = kargs.get("cols_x", [])

    # output 
    res = {}   # used to collect the statistics for later use

    X, y, fset, lset = toXY(df, cols_x=cols_x, cols_y=cols_y, scaler=None, perturb=False)
    res['labels'] = uniq_labels = np.unique(y)
    
    # special case: binary
    isSimpleBinary = False
    lmap = {}
    if len(uniq_labels) == 2: 
        lmap = {1: '+', 0: '-', -1: '-'}
        if len(cols_y) == 1: 
            isSimpleBinary = True

    vmin, vmax = np.min(X), np.max(X)
    print("[analysis] min(X): {}, max(X): {}".format(vmin, vmax))
    # print("... dim(X): {}, dim(y): {}".format(X.shape, y.shape))
    res['Xmin'], res['Xmax'] = vmin, vmax

    for label in uniq_labels: 
        polarity = lmap.get(label, '?')
        print("... column-wise mean ({}):\n{}\n".format(polarity, 
                   list(zip(fset, np.mean(X[y==label, :], axis=0)))))

    print("... column-wise mean overall:\n{}\n".format(list(zip(fset, np.mean(X, axis=0)))))

    # row_mean = np.mean(X, axis=1)
    # print("... n(row-wise 0): {}".format(np.sum(row_mean == 0)))

    # look into "trival vectors" such as zero vectors
    if isSimpleBinary: 
        col_label = cols_y[0]
        pos_label = kargs.get('pos_label', 1)
        neg_label = the_other_label(uniq_labels, given=pos_label)[0]

        # assume that the positive is encoded as 1
        df_pos = df[df[col_label]==pos_label]
        df_neg = df[df[col_label]==neg_label]
    
        df_zeros = df.loc[(df.T == 0).all()]
        print("... found n={} zero vectors overall (which cannot be normalized; troubles in computing correlation)".format(df_zeros.shape[0]))
        df_pos_zeros = df_pos.loc[(df_pos.T == 0).all()]
        df_neg_zeros = df_neg.loc[(df_neg.T == 0).all()]
        print("... N(pos): {}, n(pos, 0): {}, ratio: {}".format(df_pos.shape[0], df_pos_zeros.shape[0], df_pos_zeros.shape[0]/df_pos.shape[0]))
        print("... N(neg): {}, n(neg, 0): {}, ratio: {}".format(df_neg.shape[0], df_neg_zeros.shape[0], df_neg_zeros.shape[0]/df_neg.shape[0]))

    return res

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
        if len(y.shape) == 2: 
            y = y.reshape( (y.shape[0],) )
        sizes = collections.Counter(y)
        
    return sizes # label/col -> sample size

def summarize_dict(d, topn=15, sort_=True): 
    if topn != 0 or sort_: 
        import operator
        d = sorted(d.items(), key=operator.itemgetter(1))
    for k, v in d[:topn]: 
        print(f"[{k}] -> {v}")
    return

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

def analyze_values(df, cols=[], verbose=1, topn=10): 
    if not cols: cols = df.columns.values 

    adict = {}
    for i, col in enumerate(cols): 
        mv = collections.Counter(df[col]).most_common(topn)
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

def balance_by_downsampling2(df, cols_x=[], cols_y=[], method='medican', majority_max=3, verify=1):
    from data_processor import toXY
    
    dim0 = df.shape
    X, y, fset, lset = toXY(df, cols_x=cols_x, cols_y=cols_y, scaler=None, perturb=False)
    X, y = balance_by_downsampling(X, y, method=method, majority_max=majority_max)

    # reassemble 
    dfX = DataFrame(X, columns=fset)
    dfY = DataFrame(y, columns=lset)
    df = pd.concat([dfX, dfY], axis=1)

    if verify: 
        print("(balance_by_downsampling2) dim(df): {} -> {}".format(dim0, df.shape))
        assert df.shape[1] == dim0.shape[1]
        assert df.shape[0] <= dim0.shape[0]

    return df

def balance_by_downsampling(X, y, method='median', majority_max=3): 
    """
    Params
    ------
        majority_max: The sample size of the majority class can be at most this multiple (of the minority class sample size)
                      
                        e.g. suppose that majority_class = 3, then the majority class sample size is at most 3 times as many as 
                             the minority-class sample size

    """
    import pandas as pd
    import collections
    from sklearn.utils import resample
    
    nf = X.shape[1]
    labels = np.unique(y)
    label_dim = nf # the index for the label  
    lcnt = collections.Counter(y) # label counts
    print("(balance_by_downsampling) dim(X): {}, nl: {}, labels: {} | nf={}".format(X.shape, len(labels), labels, nf))

    lastn = 1
    Nmin = lcnt.most_common()[:-lastn-1:-1][0][1]
    Nmax = lcnt.most_common(1)[0][1]
    
    if len(y.shape) == 1: # 1-D array 
        assert X.shape[0] == len(y)
        X = np.hstack([X, y[:, None]])
    else: 
        assert X.shape[0] == y.shape[0]
        X = np.hstack([X, y])

    print("... After merging (X, y) => dim(X): {}".format(X.shape))
    
    ###########
    ts = DataFrame(X)
    if method.startswith('med'): # median
        Ncut = int(np.median([c for l, c in lcnt.items()]))
    elif method.startswith('multi'):  # multiple
        Ncut = Nmin * majority_max

    ###########
    tx = []
    for label in labels: 
        tsl = ts[ts[label_dim]==label]
        if tsl.shape[0] > Ncut:
            tx.append(tsl.sample(n=Ncut))
        else: 
            #if not tsl.empty: 
            tx.append(tsl) 

    if len(tx) > 0: 
        ts = pd.concat(tx, ignore_index=True) 

    # print("[balance] dim(ts): {} | nf-1: {}, label_dim: {}".format(ts.shape, nf-1, label_dim))

    # separate ts into (X, y)
    X = ts.iloc[:,:nf].values
    y = ts.iloc[:,nf].values

    return (X, y)

def balance_data_incr(df, df_extern, n_samples=-1, col='test_result_loinc_code', labels=[], verbose=1, **kargs):
    """
    Similar to balance_data() but expect df_extern to be huge (to a degree that may not fit into main memory)
    """
    # some preliminary check here

    return dproc.balance_data_incr(df, df_extern, n_samples=n_samples, col=col, labels=labels, verbose=verbose, **kargs)

def balance_data(df, df2=None, n_samples=-1, col='test_result_loinc_code', labels=[], verbose=1): 
    """
    Make class-balanced training data by balancing sample sizes.

    Input
    -----
    df: Input dataframe whose columns contain both features/variables and class labels

    """
    return dproc.balance_data(df, df2=df2, n_samples=n_samples, col=col, labels=labels, verbose=verbose)


########################################
def plot_barh(perf, metric='fmax', **kargs): 
    return evaluate.plot_barh(perf, metric=metric, **kargs)

def compare_col_values(df, cols, n=10, mode='sampling', verbose=2, **kargs):   
    from loinc import compare_6parts

    random_state = kargs.get('random_state')
    include_6parts = kargs.get('include_6parts', False)
    include_mtrt = kargs.get('include_mtrt', False)
    df_loinc = kargs.get('df_loinc', None)
    col_code = kargs.get('col_code', LoincTSet.col_code) # test_result_loinc_code
    dehyphen = kargs.get('dehyphenate', True)

    if not LoincTSet.col_code in cols: cols = [col_code, ] + list(cols)

    # all target LOINC codes
    codes = df[col_code].unique()

    df = col_values2(df, cols=cols, n=n, mode=mode, random_state=random_state, keep='last')
    if mode.startswith(('s', 'r')): assert df.shape[0] <=n, "dim(df): {}".format(df)

    D6p = {}
    if include_6parts: 
        if df_loinc is None: df_loinc = load_loinc_table(dehyphenate=dehyphen)
        D6p = compare_6parts(df_loinc, codes=codes, verbose=1)
        print("(compare_col_values) attributes for a given code:\n{}\n".format( list(next(iter(D6p.values())).keys()) ))

    Dm = {}
    if include_mtrt: 
        col_mval = LoincMTRT.col_value # MTRT
        col_mkey = LoincMTRT.col_key    # loinc codes in the mtrt table
        df_mtrt = LoincMTRT.load_table(dehyphenate=dehyphen, one_to_one=True)
         
        for code in codes:
            dfe = df_mtrt[df_mtrt[col_mkey] == code]
            if not dfe.empty:  
                Dm[code] = dfe[col_mval].iloc[0]

    adict = {col:[] for col in cols}
    codes_undef = set([]) 
    for i, (r, row) in enumerate(df.iterrows()): 
        code = row[col_code]
        msg = "[{}] Code: {} | iloc={}, cols={}\n".format(i+1, code, r, cols)
        for col in cols: 
            adict[col].append(row[col])
            msg += "    + {}: {}\n".format(col, row[col])

        # -------------------------------------------
        if code in Dm: 
            msg += "    + {}: {}\n".format('MTRT', Dm[code])

        if len(D6p) > 0: 
            cols_6p = LoincTable.cols_6p   # ['COMPONENT', 'PROPERTY', 'TIME_ASPCT', 'SYSTEM', 'SCALE_TYP', 'METHOD_TYP', ]
            
            # it's possible that the loinc code is not found in the standard LOINC table
            if code in D6p: 
                for col, val in D6p[code].items(): 
                    if verbose == 1 and not col in cols_6p: continue
                    msg += "      ++ {}: {}\n".format(col, val)
            else: 
                if verbose > 1: msg += "\n!!! Code {} not found in LOINC table. Obsolete?\n".format(code)
                # n_undef += 1
                codes_undef.add(code)


        if verbose: 
            print(msg)
    if len(codes_undef) > 0: highlight("Found n={} undefined codes:\n{}\n".format(len(codes_undef), list(codes_undef)))
    return adict

def inspect_col_values(df, cols, mode='unique', verbose=1):

    adict = {}
    msg = ""
    for i, col in enumerate(cols): 
        adict[col] = col_values(df, col=col, mode=mode)
        msg += f"[{i}] {col}\n"
        msg += "       + n={}: {{ {} }}\n".format(len(adict[col]), adict[col])
    if verbose: 
        print(msg)
    
    return adict

def analyze_loinc_table(**kargs):
    # from analyzer import load_loinc_table, sample_loinc_table
    import loinc as lc

    cols_6p = lc.LoincTable.cols_6p # ["COMPONENT","PROPERTY","TIME_ASPCT","SYSTEM","SCALE_TYP","METHOD_TYP"]   # CLASS
    col_code = 'LOINC_NUM'

    df_loinc = load_loinc_table(dehyphenate=True)
    print("(loinc_table):\n{}\n".format(list(df_loinc.columns.values)))

    # assert sum(1 for col in cols_6p if not col in df_loinc.columns) == 0  # ... ok
    
    codes_src = set(df_loinc[col_code].values)
    print("(loinc_table) Number of unique loinc codes: {}".format( len(codes_src)) )

    ### Q: what are all the possible values for earch of the the 6-part represention of LOINC codes? 
    inspect_col_values(df_loinc, cols_6p, mode='unique', verbose=1)

    ### long vs short names
    cohort = 'hepatitis-c'
    df_perf = load_performance(input_dir='result', cohort=cohort)
    codes_low_sz = df_perf.loc[df_perf['mean'] < 0]['code'].values

    D_loinc = lc.compare_short_long_names(df_loinc, codes=codes_low_sz, n_display=30, verbose=1)

    ### mtrt vs long name 
    # df_mtrt = LoincMTRT.load_table()
    D_mtrt = lc.compare_longname_mtrt(n_display=30, codes=codes_low_sz)

    print("[analysis] Can component and system alone uniquely identify most LOINC codes?")

    Dg = lc.group_by(df_loinc=df_loinc, cols=['COMPONENT', 'SYSTEM',])
    

    return

def compare_test_with_6parts(codes=[], df=None, df_loinc=None, 
        col_code='test_result_loinc_code', n_samples=-1, target_test_cols=[], **kargs):
    import loinc as lc

    verbose = kargs.get('verbose', 1)
    cohort = kargs.get('cohort', "hepatitis-c")
    processed = kargs.get('processed', True)
    if df is None: 
        df = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=True, processed=processed)

    if df_loinc is None: df_loinc = load_loinc_table(dehyphenate=True)

    if not target_test_cols: target_test_cols = ['test_order_name', 'test_result_name']

    if not codes: 
        ucodes = df[col_code].unique()
        if n_samples > 0: 
            codes = np.random.choice(ucodes, min(n_samples, len(ucodes))) 
        else: 
            codes = ucodes  # all unique codes
    if verbose: print("(compare_test_with_6parts) Targeting n={} codes:\n{} ... #\n".format(len(codes), codes[:100]))

    D = lc.compare_6parts(df_loinc, codes=codes, verbose=1)  # n_samples/-1
        
    for code in codes: 
        if code in D: 
            dfc = df.loc[df[col_code] == code]
            assert not dfc.empty

            row = dfc.sample(n=1)
            for col in target_test_cols: 
                D[code][col] = row[col].iloc[0]

    ######################################################
    if verbose:
        target_codes = list(D.keys())
        cols_key = ['code', ]
        cols_6p = ['COMPONENT', 'SYSTEM']
        cols_loinc_names = ['SHORTNAME', 'LONG_COMMON_NAME'] 
        target_cols = target_test_cols + cols_6p + cols_loinc_names
        adict = {col: [] for col in (cols_key+target_cols)}
            
        # display contents in terms of a dataframe
        for code in target_codes: 
            adict['code'].append(code)
            for tc in target_cols: 
                adict[tc].append(D[code][tc]) 
        
        df = DataFrame(adict, columns=cols_key+target_cols)

        cols_part0 = cols_key + target_test_cols
        cols_part1 = cols_key + cols_6p + ['SHORTNAME', ]
        cols_part2 = cols_key + cols_loinc_names
        print("(compare_test_with_6parts) {} (part0):\n{}\n".format(target_cols, 
            tabulate(df[cols_part0], headers='keys', tablefmt='psql'))) # df[cols_part1].to_string(index=False))) # tabulate(df, headers='keys', tablefmt='psql')
        
        print("... {} (part1):\n{}\n".format(target_cols,
            tabulate(df[cols_part1], headers='keys', tablefmt='psql')))
        print("... {} (part2):\n{}\n".format(target_cols,
            tabulate(df[cols_part2], headers='keys', tablefmt='psql')))
        
    return D

#########################################################################
# --- Analysis Functions --- # 

def analyze_loinc_values(ccmap, cols=[], verbose=1, col_code='LOINC_NUM'):
    """

    Related
    -------
    analyze_feature_values
    """
    return  analyze_by_values(ccmap, cols=cols, verbose=verbose, col_code=col_code)
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
    from transformer import canonicalize, dehyphenate
    assert isinstance(ccmap, dict)

    df_loinc = load_loinc_table(dehyphenate=True) # input_dir/'LoincTable', input_file/'LoincTable.csv', sep/','
    # dehyphenate(df_loinc, col=col_code)  # inplace

    if not col_code: col_code = LoincTable.col_key  # 'LOINC_NUM'
    
    target_cols = cols
    if len(cols) == 0: 
        
        text_cols = LoincTable.text_cols #  ['LONG_COMMON_NAME', 'SHORTNAME', 'RELATEDNAMES2', 'STATUS_TEXT']
        property_cols = LoincTable.p6 + ['CLASS', ] # [ 'COMPONENT', 'PROPERTY', 'TIME_ASPCT', 'SYSTEM', 'METHOD_TYP', 'SCALE_TYP', 'CLASS', ]  
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

def analyze_data_set(**kargs):

    cohort = 'hepatitis-c'
    verbose = 1
    # andromeda-pond-hepatitis-c-balanced.csv has illed-formed data
    col_target = "test_result_loinc_code"
    col_tag = 'medivo_test_result_type'
    token_default = 'unknown'

    df = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=True, processed=True)
    N0 = df.shape[0]
    # ... if canonicalized <- True => call canonicalize(): fill n/a + dehyphenate + replace_values + trim_tail + fill others (non-target classes)
    
    # summarize_dataframe(df, n=1)
    # print("-" * 50 + "\n")

    # target_cols = ['meta_sender_name', 'test_order_name', 'test_specimen_type', ]
    # print(f"(analyze_data_set) Inspecting {target_cols} ...")
    # inspect_col_values(df, target_cols, mode='unique', verbose=1)

    # --- analyze the relationship between test properties (e.g. test_order_name) and LOINC code properties

    print("[analysis] Now compare test_order_name, test_result_name ... etc.")
    # compare_test_with_6parts(df=df, n_samples=30)

    compare_test_with_6parts(df=df, n_samples=20, target_test_cols=['test_order_name', 'test_result_name', 'medivo_test_result_type'])

    # --- MTRT vs LOINC codes 
    # Given MTRT
    df_mtrt = df.loc[~df[col_tag].isnull()]
    N_mtrt = df_mtrt.shape[0]
    print("[analysis] N_mtrt={}, n(total)={} | ratio: {}".format(N_mtrt, N0, N_mtrt/(N0+0.0)))
    df_mtrt_loinc = df_mtrt.loc[~df_mtrt[col_target].isnull()]
    df_mtrt_no_loinc = df_mtrt.loc[df_mtrt[col_target].isnull()]
    df_mtrt_TO = df_mtrt.loc[~df_mtrt['test_order_name'].isnull()]
    df_mtrt_TR = df_mtrt.loc[~df_mtrt['test_result_name'].isnull()]
    df_mtrt_TC = df_mtrt.loc[~df_mtrt['test_result_comments'].isnull()]
    print("... N(mtrt & loinc) ={}, N={} | ratio: {}".format(df_mtrt_loinc.shape[0], N_mtrt, df_mtrt_loinc.shape[0]/(N_mtrt+0.0)))
    print("... N(mtrt & ~loinc)={}, N={} | ratio: {}".format(df_mtrt_no_loinc.shape[0], N_mtrt, df_mtrt_no_loinc.shape[0]/(N_mtrt+0.0) ))
    ##########################
    print("... N(mtrt & TO) ={}, n(mtrt)={} | ratio: {}".format(df_mtrt_TO.shape[0], N_mtrt, df_mtrt_TO.shape[0]/(N_mtrt+0.0) ))
    print("... N(mtrt & TR) ={}, n(mtrt)={} | ratio: {}".format(df_mtrt_TR.shape[0], N_mtrt, df_mtrt_TR.shape[0]/(N_mtrt+0.0) ))
    print("... N(mtrt & TC) ={}, n(mtrt)={} | ratio: {}".format(df_mtrt_TC.shape[0], N_mtrt, df_mtrt_TC.shape[0]/(N_mtrt+0.0) ))


    # --- MTRT vs result comments
    print("[anaysis] When MTRT is missing, what else do we usually have among test_order_name, test_result_name, *test_result_comments?")
    df_no_mtrt = df.loc[df[col_tag].isnull()]   
    N_mtrt_none = df_no_mtrt.shape[0]
    df_no_mtrt_TO = df_no_mtrt.loc[~df_no_mtrt['test_order_name'].isnull()]
    df_no_mtrt_TR = df_no_mtrt.loc[~df_no_mtrt['test_result_name'].isnull()]
    df_no_mtrt_TC = df_no_mtrt.loc[~df_no_mtrt['test_result_comments'].isnull()]
    print("... N(mtrt_none & TO) ={}, n(mtrt_none)={} | ratio: {}".format(df_no_mtrt_TO.shape[0], N_mtrt_none, df_no_mtrt_TO.shape[0]/(N_mtrt_none+0.0) ))
    print("... N(mtrt_none & TR) ={}, n(mtrt_none)={} | ratio: {}".format(df_no_mtrt_TR.shape[0], N_mtrt_none, df_no_mtrt_TR.shape[0]/(N_mtrt_none+0.0) ))
    print("... N(mtrt_none & TC) ={}, n(mtrt_none)={} | ratio: {}".format(df_no_mtrt_TC.shape[0], N_mtrt_none, df_no_mtrt_TC.shape[0]/(N_mtrt_none+0.0) ))
    # ... [insight]
    #      1. test_result_comments occur about just as equally among rows with and without MTRT 
    #         (intuitively, those without MTRT would have lower missing rate for test_result_comments but it's not quite true)
    #     
    
    # --- Level of difficulty
    df, adict = label_by_types(df=df, cohort=cohort, transformed_vars_only=False)
    msg = ''
    for k, row_ids in adict.items(): 
        # msg = 'LOINC(1), MTRT(1)'
        print("... Type({}): N={}".format(k, len(row_ids)))

    # --- group by 
    # dproc.group_by(df, cols=['test_result_code', 'test_result_name',], verbose=1, n_samples=10)

    # ... which groups correspond to which LOINC codes? 
    # dproc.map_group_to_label(df, cols=['test_result_code', 'test_result_name',], verbose=1, n_samples=-1)

    return df

def analyze_hard_cases(**kargs):
    """
    Analyze hard cases (loinc codes), for which classifier-based approach performed poorly. 
    Hard cases include those with low sample sizes. 

    """ 
    from utils_sys import size_hashtable
    
    cohort = 'hepatitis-c'
    col_target = 'test_result_loinc_code'
    categories = ['easy', 'hard', 'low']  # low: low sample size
    ccmap = label_by_performance(cohort=cohort, categories=categories)
    N = size_hashtable(ccmap)
    N_easy = len(ccmap['easy'])
    N_hard = len(ccmap['hard'])
    N_low = len(ccmap['low'])
    print("[analysis] N(codes; cohort={}): {}".format(cohort, N))
    print("...        n(easy): {} | ratio: {}".format(N_easy, N_easy/(N+0.0)))
    print("...        n(hard): {} | ratio: {}".format(N_hard, N_hard/(N+0.0)))
    print("...        n(low):  {} | ratio: {}".format(N_low,  N_low/(N+0.0)))

    # a dictionary of LOINC table dataframes indexed by analysis categories (e.g. easy, hard)
    highlight("[analysis] 6 Parts")
    compare_test_with_6parts(code=ccmap['low'], n_samples=30, target_test_cols=['test_order_name', 'test_result_name']) 

    # --- Get text data for the hard cases from 'test_order_name', 'test_result_name', 'test_result_comments', ... 
    codes_low_sz = ccmap['low']
    dfp = load_src_data(cohort='hepatitis-c', warn_bad_lines=False, canonicalized=True, processed=True)
    dim0 = dfp.shape
    df_lsz = dfp.loc[dfp[col_target].isin(codes_low_sz)]
    print("[analysis] dim(df)<low sample size>: {} | dim(df)<original>: {}".format(df_lsz.shape, dim0))

    highlight("[analysis] T-attributes, which ones can predict which LOINC parts?")
    # T-attributes? e.g. test_order_name, test_result_name ... 

    target_cols = ['test_order_name', 'test_result_name', 'test_result_comments', 'test_specimen_type',  'test_result_units_of_measure'] # 'panel_order_name'
    compare_col_values(df_lsz, target_cols, n=100, mode='sampling', 
        include_6parts=True, include_mtrt=True, verbose=2, random_state=53)

    # --- 
    highlight("[analysis] Some features only have values in a subset of rows and may not be consistent?")
    # e.g. test_specimen_type of only has value in a subset of the rows associated with a LOINC code? 
    
    col_code = LoincTSet.col_code
    # --- LOINC table attributes
    col_ln, col_sn = LoincTable.long_name, LoincTable.short_name
    col_lkey = LoincTable.col_key
    ##################################
    # --- Loinc to MTRT attributes (from Leela)
    col_mval = LoincMTRT.col_value
    col_mkey = LoincMTRT.col_key    # loinc codes in the mtrt table

    target_cols = ['test_order_name', 'test_result_name', 'test_specimen_type',   ]
    adict = {}
    n_hetero = 0
    for code, dfe in df_lsz.groupby([col_code, ]):
        adict[code] = {col:set([]) for col in target_cols}
        
        nvalues = set([])
        for col in target_cols: 
            # do these column always have consistent values
            vals = dfe[col].unique()
            adict[code][col] = vals
            nvalues.add(len(vals))

        if len(nvalues) > 1: 
            msg = "... Code: {} has multiple values observed in some attributes (e.g. >=2 test_order_name) ...\n".format(code)
            for col, values in adict[code].items(): 
                msg += "    + {} (nv={})\n".format(col, len(values))
                for i, v in enumerate(values): 
                    msg += "      ++ {}\n".format(v)
            print(msg)
            n_hetero += 1
    highlight("... Found n={} codes associated attributes having > values (e.g. same code, >1 values intest_order_name)".format(n_hetero))
            
    return

def analyze_feature_values():
    """

    Related
    -------
    analyze_loinc_values()
    """
    from utils_sys import size_hashtable

    # If we decide to filter the sample by disease cohort or type ... 
    # ---------------------------------------------------------- 
    cohort = 'hepatitis-c'
    col_target = 'test_result_loinc_code'
    categories = ['easy', 'hard', 'low']  # low: low sample size
    ccmap = label_by_performance(cohort=cohort, categories=categories)
    N = size_hashtable(ccmap)
    N_easy = len(ccmap['easy'])
    N_hard = len(ccmap['hard'])
    N_low = len(ccmap['low'])
    # ----------------------------------------------------------

    target_cols = ['test_order_name', 'test_result_name', 'test_specimen_type',   ]
    # target_cols = ['test_order_name', 'TestOrderMapJW', 'TOPredictedComponentJW', 'TOMatchDistComponentJW',]
    
    for col in target_cols: 
        input_file = f'{col}-sdist-vars.csv'
        df = load_generic(input_file=input_file, sep=',')
        print("(analysis) col: {} dim(input_df): {} ...".format(col, df.shape))

        if df is not None and not df.empty: 
            cols_template = [col, 'MapJW', 'PredictedComponentJW', 'MatchDistComponentJW', ]
           
            target_cols = []
            for col in df.columns: 
                tMatched = False
                for ct in cols_template: 
                    if col.find(ct) >= 0: 
                        tMatched = True
                        break
                if tMatched: 
                    target_cols.append(col)
            # ... target_cols determined 
            compare_col_values(df, target_cols, n=50, mode='sampling', verbose=1, random_state=53)


    return

def analyze_missing_cases(df, **kargs):
    pass

########################################################################
# --- Example Usage and Test Cases --- #

def demo_performance(**kargs): 
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    print("> before y:\n{}\n".format(y))
    y = encode_labels(y, pos_label=1)

    print("> after  y:\n{}\n".format(y))

    scores = eval_performance(X, y, model=None, cv=5, random_state=53)
    print("> average: {}, std: {}".format(np.mean(scores), np.std(scores)))

    return scores 

def demo_performance_stats(**kargs):
    categories = ['easy', 'hard', 'low']  # low: low sample size

    ccmap = label_by_performance(cohort='hepatitis-c', categories=categories)

    # a dictionary of LOINC table dataframes indexed by analysis categories (e.g. easy, hard)
    D = analyze_by_values(ccmap, verbose=1) #  

    return 

def demo_stratify(**kargs): 
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
    tProcessed = False

    df_perf = load_performance(input_dir='result', cohort=cohort)

    # ... list of 2-tuples: [(code<i>, count<i>) ... ]
    max_size = np.max(df_perf['n_pos'].values)
    print("(t_stratify) control class sample size to be: {}".format(max_size))

    # load training data 
    # cohort = 'loinc-hepatitis-c'

    # load source data
    # ts0 = load_data(input_file='andromeda-pond-hepatitis-c.csv', warn_bad_lines=False)
    ts0 = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=True, processed=tProcessed)
    # ts0 = ts0.drop_duplicates(keep='last')  # drop duplicates 
    # ts0 = lc.canonicalize(ts0, col_target=col_target, token_missing=token_default)
    N0 = ts0.shape[0]

    print("(t_stratify) dim(ts0): {}".format(ts0.shape))  
    loinc_set = codes0 = ts0[col_target].unique()
    codes0_subset = np.random.choice(codes0, 30)
    print("(t_stratify) example codes (source):\n{}\n".format(list(codes0_subset)))

    # load source data
    ts_ctrl = load_src_data(input_file='andromeda_pond-10p.csv', warn_bad_lines=False)
    # ts_ctrl = ts_ctrl.drop_duplicates(keep='last')  # drop duplicates
    # ts_ctrl = lc.canonicalize(ts_ctrl, col_target=col_target, token_missing=token_default, target_labels=loinc_set)
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

def demo_loinc(**kargs):
    from transformer import dehyphenate, trim_tail, replace_values

    df = load_src_data(canonicalized=False, processed=False)   # cohort='hepatitis-c'cohort='hepatitis-c'

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

def test_indv(**kargs):
    
    # --- Feature value range analysis
    X = np.array([ [0, 1, 0, 1], 
                   [0, 0, 0, 0], 
                   [1, 1, 0, 1], 
                   [0, 0, 0, 0], 
                   [1, 1, 1, 1]
                       ])
    y = np.array([1, 0, 1, 0, 1])
    fset = ['x1', 'x2', 'x3', 'x4', 'x5']

    det_range2(X, y, fset=fset, pos_label=1)

    return

def test(**kargs):
    subjects = [ 'hard', ] # {'table', 'ts'/'data', 'hard', 'feature'/'vars'}
    verbose = 1

    for subject in subjects: 

        ### Analyze training data 
        if subject.startswith(('d', 'ts')): 
            analyze_data_set()

        ### Stratigy training data 
        # demo_stratify()   # ... ok 

        ### Predictive performance
        # demo_performance(**kargs)
        # demo_performance_stats(**kargs)

        ### LOINC codes analysis
        # demo_loinc(**kargs)

        if subject.startswith( 'tab'): # data, ts
            analyze_loinc_table()

        if subject.startswith('hard'):  # hard cases 
            analyze_hard_cases(verbose=verbose) 

        if subject.startswith( ('feat', 'var') ):
            analyze_feature_values()

    return

if __name__ == "__main__":
    # test()

    test_indv()