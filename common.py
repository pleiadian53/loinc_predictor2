"""


Reference
---------
    1. datasink: A Pipeline for Large-Scale Heterogeneous Ensemble Learning
                 <link> https://github.com/shwhalen/datasink
                 
"""
import os, time
import random
import numpy as np
import re, collections, glob, random
from numpy import argmax, argmin, argsort, corrcoef, mean, nanmax, sqrt, triu_indices_from, where
from pandas import DataFrame, concat, read_csv
from scipy.io.arff import loadarff
import sklearn.metrics

import warnings
warnings.filterwarnings("ignore")
# with warnings.catch_warnings():
#         warnings.filterwarnings("ignore",category=UserWarning)
#         warnings.filterwarnings("ignore", category=DeprecationWarning)


# def levenshtein(a, b, mx=-1):    
#     def result(d): return d if mx < 0 else False if d > mx else True
 
#     if a == b: return result(0)
#     la, lb = len(a), len(b)
#     if mx >= 0 and abs(la - lb) > mx: return result(mx+1)
#     if la == 0: return result(lb)
#     if lb == 0: return result(la)
#     if lb > la: a, b, la, lb = b, a, lb, la
 
#     cost = array('i', range(lb + 1))
#     for i in range(1, la + 1):
#         cost[0] = i; ls = i-1; mn = ls
#         for j in range(1, lb + 1):
#             ls, act = cost[j], ls + int(a[i-1] != b[j-1])
#             cost[j] = min(ls+1, cost[j-1]+1, act)
#             if (ls < mn): mn = ls
#         if mx >= 0 and mn > mx: return result(mx+1)
#     if mx >= 0 and cost[lb] > mx: return result(mx+1)
#     return result(cost[lb])

def perturb(X, cols_x=[], cols_y=[], lower_bound=0, alpha=100.):
    def add_noise():
        min_nonnegative = np.min(X[np.where(X>lower_bound)])
        
        Eps = np.random.uniform(min_nonnegative/(alpha*10), min_nonnegative/alpha, X.shape)

        return X + Eps
    # from pandas import DataFrame

    if isinstance(X, DataFrame):
        from data_processor import toXY
        X, y, fset, lset = toXY(X, cols_x=cols_x, cols_y=cols_y, scaler=None, perturb=False)
        X = add_noise(X)
        dfX = DataFrame(X, columns=fset)
        dfY = DataFrame(y, columns=lset)
        return pd.concat([dfX, dfY], axis=1)

    X = add_noise()
    return X

def scale(X, scaler=None, **kargs):
    from sklearn import preprocessing
    if scaler is None: 
        return X 

    if isinstance(scaler, str): 
        if scaler.startswith(('standard', 'z')): # z-score
            std_scale = preprocessing.StandardScaler().fit(X)
            X = std_scale.transform(X)
        elif scaler.startswith('minmax'): 
            minmax_scale = preprocessing.MinMaxScaler().fit(X)
            X = minmax_scale.transform(X)
        elif scaler.startswith("norm"): # normalize
            norm = kargs.get('norm', 'l2')
            copy = kargs.get('copy', False)
            X = preprocessing.Normalizer(norm=norm, copy=copy).fit_transform(X)
    else: 
        try: 
            X = scaler.transform(X)
        except Exception as e: 
            msg = "(scale) Invalid scaler: {}".format(e)
            raise ValueError(msg)
    return X
# --- alias
apply_scaling = scale

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelBinarizer
    
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

def ordered_sampled_without_replacement(seq, k):
    if not (0 <= k <=len(seq)):
        raise ValueError('Required that 0 <= sample_size <= population_size')

    numbersPicked = 0
    for i,number in enumerate(seq):
        prob = (k-numbersPicked)/(len(seq)-i)
        if random.random() < prob:
            yield number
            numbersPicked += 1

def five_number(x): 
    # five number summary of a sequence x
    from numpy import percentile
    # from numpy.random import rand

    # generate data sample
    # data = rand(1000)

    # calculate quartiles
    quartiles = percentile(x, [25, 50, 75])
    # calculate min/max
    x_min, x_max = np.min(x), np.max(x)

    return (x_min, quartiles[0], quartiles[1], quartiles[2], x_max)

def argsortbest(x):
    return argsort(x) if greater_is_better else argsort(x)[::-1]

def average_pearson_score(x):
    if isinstance(x, DataFrame):
        x = x.values

    # compute pairwise correlations
    rho = corrcoef(x, rowvar = 0) # rowvar: False => each column represents a variable, while the rows contain observations
    return mean(abs(rho[triu_indices_from(rho, 1)])) # average the off-diag corr scores 


def get_best_performer(df, one_se = False):
    if not one_se:
        return df[df.score == best(df.score)].head(1)
    se = df.score.std() / sqrt(df.shape[0] - 1)
    if greater_is_better:
        return df[df.score >= best(df.score) - se].head(1)  # tolerance by one SE
    return df[df.score <= best(df.score) + se].head(1)


def confusion_matrix_fpr(labels, predictions, false_discovery_rate = 0.1):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, predictions)
    max_fpr_index = where(fpr >= false_discovery_rate)[0][0]
    print (sklearn.metrics.confusion_matrix(labels, predictions > thresholds[max_fpr_index]))


def fmax_score(labels, predictions, beta = 1.0, pos_label = 1):
    """
        Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
        Manning, C. D. et al. (2008). Evaluation in Information Retrieval. In Introduction to Information Retrieval. Cambridge University Press.

    Memo
    ---- 
    1. precision and recall tradeoff doesn't take into account true negative

    """
    precision, recall, threshold = sklearn.metrics.precision_recall_curve(labels, predictions, pos_label)

    # the general formula for positive beta
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    # if beta == 1, then this is just f1 score, harmonic mean between precision and recall 
    # i = np.nanargmax(f1)

    # return (f1[i], threshold[i])
    return nanmax(f1)

def fmax_score_threshold(labels, predictions, beta = 1.0, pos_label = 1):
    """

    Memo
    ---- 
    1. precision and recall tradeoff doesn't take into account true negative
       
       precision: Precision values such that element i is the precision of predictions with score >= thresholds[i] and the last element is 1. 
       recall: Decreasing recall values such that element i is the recall of predictions with score >= thresholds[i] and the last element is 0.

    2. example 

    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> precision, recall, thresholds = precision_recall_curve(
    ...     y_true, y_scores)
    >>> precision  
    array([0.66666667, 0.5       , 1.        , 1.        ])
    >>> recall
    array([1. , 0.5, 0.5, 0. ])
    >>> thresholds
    array([0.35, 0.4 , 0.8 ])

    precision[1] = 0.5, for any prediction >= thresholds[1] = 0.4 as positive (assuming that pos_label = 1)

    """
    import numpy as np
    precision, recall, threshold = sklearn.metrics.precision_recall_curve(labels, predictions, pos_label)

    # the general formula for positive beta
    # ... if beta == 1, then this is just f1 score, harmonic mean between precision and recall 
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    i = np.nanargmax(f1)  # the position for which f1 is the max 
    th = threshold[i] if i < len(threshold) else 1.0    # len(threshold) == len(precision) -1 
    # assert f1[i] == nanmax(f1)
    return (f1[i], th)

def fmax_precision_recall_scores(labels, predictions, beta = 1.0, pos_label = 1):
    import numpy as np

    ret = {}
    precision, recall, threshold = sklearn.metrics.precision_recall_curve(labels, predictions, pos_label)

    # the general formula for positive beta
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    i = np.nanargmax(f1)
    th = threshold[i] if i < len(threshold) else 1.0    # len(threshold) == len(precision) -1 

    ret['id'] = i 
    ret['threshold'] = th
    ret['precision'] = precision[i] # element i is the precision of predictions with score >= thresholds[i]
    ret['recall'] = recall[i]
    ret['f'] = ret['fmax'] = f1[i]
    
    return ret  # key:  id, precision, recall, f/fmax


def load_arff(filename):
    return DataFrame.from_records(loadarff(filename)[0])


def load_arff_headers(filename):
    dtypes = {}
    for line in open(filename):
        if line.startswith('@data'):
            break
        if line.startswith('@attribute'):
            _, name, dtype = line.split()
            if dtype.startswith('{'):
                dtype = dtype[1:-1]
            dtypes[name] = set(dtype.split(','))
    return dtypes

# datasink original
def load_properties0(dirname):
    properties = [_.split('=') for _ in open(dirname + '/weka.properties').readlines()]
    d = {}
    for key, value in properties:
        d[key.strip()] = value.strip()
    return d

def load_properties(dirname, config_file='config.txt'):
    """
    Configuration parser. 

    Memo
    ----
    1. datasink uses weka.properties instead of config.txt
    """

    # [todo] add remove comments
    properties = [line.split('=') for line in open(dirname + '/%s' % config_file).readlines() 
                    if len(line) > 0 and not line.strip().startswith('#')]

    # [todo] more flexible parser tolerant of trailing newlines
    # lines = []
    # for line in open(dirname + '/%s' % config_file).readlines(): 
    #     elems = line.split('=')
   
    d = {}
    for key, value in properties:
        d[key.strip()] = value.strip()
    return d

def readAll(path, dataset='bp', file_type='predictions', reconstructed_testset=True, ext='csv.gz', exception_=True): 
    """
    Read and combine the training data (matching the string specified by 'dataset') if dtype = 'validation'; otherwise, 
    read and combine the test data. 

    """
    method_id, indices = resolve(path, dataset=dataset, file_type='predictions', 
        reconstructed_testset=reconstructed_testset, ext=ext, exception_=exception_)

    dfs = []
    for index in indices: 
        train_df, train_labels, test_df, test_labels = read(index, path=path, dataset=method_id)
        if file_type == 'validation': 
            dfs.append(train_df)
        else: 
            dfs.append(test_df)
    df = concat(dfs, axis = 0)
    labels = df.index.get_level_values('label').values
    return (df, labels)
def readAllIter(path, dataset='bp', file_type='predictions', reconstructed_testset=True, ext='csv.gz', exception_=True, dev_ratio=None, test_ratio=0.2, max_size=None, n_runs=10):

    method_id, indices = resolve(path, dataset=dataset, file_type=file_type, ext=ext, exception_=exception_)
    print('(readAllIter) method_id: {id}, indices: {idx}    ... (verify) #'.format(id=method_id, idx=indices))
    
    if file_type.startswith(('prior', 'posterior', )): 
        if not indices: indices = range(n_runs)

        for index in indices: 
            yield shuffle_split_reconstructed(path, method=dataset, index=index, split_number=2, dev_ratio=dev_ratio, test_ratio=test_ratio, max_size=max_size, file_type=file_type)
            
            # assuming that we use this iterator for testing (e.g. testing mean predictions), it only makes sense to return the test split
            # yield test_df, test_labels
    else: 
        for index in indices: 
            train_df, train_labels, test_df, test_labels = read(index, path=path, dataset=method_id)
            if file_type == 'validation': 
                labels = train_df.index.get_level_values('label').values
                yield train_df, labels
            elif file_type.startswith('pred'): 
                labels = test_df.index.get_level_values('label').values 
                yield test_df, labels 
            else: 
                raise ValueError('Unrecognized file_type: {dtype}'.format(dtype=file_type))

def resolve(path, dataset='bp', file_type='predictions', reconstructed_testset=True, ext='csv.gz', exception_=True): 
    method_id = dataset
    if file_type.startswith( ('tr', 'v')):
        file_type = 'validation' 

    datasets = match_exact(path=path, method=dataset, file_type=file_type, ext=ext, verify=True) 
    assert len(datasets) > 0, "(readAll) Could not find matching datasets using method_id: %s" % method_id
    if len(datasets) > 1: 
        msg = '... (verify) Found multiple datasets (n={n}) matching {id} (intended to be ambiguous?)'.format(n=len(datasets), id=method_id)
        if exception_: 
            raise ValueError(msg)
        else: 
            div(msg, symbol='%', border=1)
            print('... Found the following matching datasets:')
            for mid, indices in datasets.items(): 
                print('...... ID: {name}, indices/folds: {ids}'.format(name=mid, ids=indices))

            # greedy, choose the one with longest ID ... [todo]
            method_id, indices, _ = sorted([(mid, indices, len(mid)) for mid, indices in datasets.items()], key=lambda x: x[2], reverse=True)[0]
    else: 
        print('... (verify) Found single dataset (n=1) matching {id}'.format(id=method_id))
        indices = datasets[method_id]

    if not indices: 
        assert file_type.startswith(('prior', 'post')), "Invalid file_type to have no indices: {dtype}".format(dtype=file_type)
        # print('... (verify) Non-partitioned dataset found: file_type is either prior or posterior')

    return (method_id, indices)

def read(fold, path, dataset='bp', reconstructed_testset=True):
    if dataset == 'bp': 
        return read_fold(path, fold)
    return read_fold_reconstructed(path, fold, method=dataset)

def get_data(path, dataset='bp', fold_count=-1):

    ########################################################
    fold = 0
    if fold_count > 1: 
        fold = np.random.choice(range(fold_count), 1)[0]  # random.sample(range(fold_count), 1)[0]
    ########################################################
    train_df, train_labels, test_df, test_labels = read(fold, path, dataset=dataset)
    df = concat([train_df, test_df])
    labels = np.hstack((train_labels, test_labels))
    return df, labels

def read_fold(path, fold, shuffle=False, test_ratio=None, fold_count=5, random_state=None):
    ### validation set is a combintation of the following: 
    #   foreach classifier, combine all their bagged versions  (X.0, X.1, X.2), say bag_count = 3, this expands data horizontally, col-wise) ... (1a)
    #   do the above for all inner folds  (this expands data vertically, row-wise)  ... (1b)
    #   do (1a), (1b) for all classifiers 
    #      => this completes one outer CV fold
    if shuffle: 
        from sklearn.model_selection import train_test_split

        if random_state is None: random_state = int(time.time()+random.randint(1, 1000)+fold)

        # aggregate and then reshuffle
        dfs = []
        train_df = read_csv('%s/validation-%i.csv.gz' % (path, fold), index_col = [0, 1], compression = 'gzip')
        test_df = read_csv('%s/predictions-%i.csv.gz' % (path, fold), index_col = [0, 1], compression = 'gzip')
        train_size = train_df.shape[0]
        test_size = test_df.shape[0]
        dfs.extend([train_df, test_df])

        # aggregate 
        df = concat(dfs, axis = 0)

        labels = df.index.get_level_values('label').values
        # print('... ids: {ids}'.format(ids=df.index.get_level_values('id').values[:100]))
        
        ### train-test split
        if test_ratio is None or test_ratio <= 0: test_ratio = 1/(fold_count+0.0)

        train_df, test_df, train_labels, test_labels = train_test_split(df, labels, test_size=test_ratio, 
            shuffle=True, stratify=labels, random_state=random_state)  # random_state
        # assert all(train_df.index.get_level_values('label').values == train_labels)
    else:
        train_df        = read_csv('%s/validation-%i.csv.gz' % (path, fold), index_col = [0, 1], compression = 'gzip')
        test_df         = read_csv('%s/predictions-%i.csv.gz' % (path, fold), index_col = [0, 1], compression = 'gzip')
        train_labels    = train_df.index.get_level_values('label').values
        test_labels     = test_df.index.get_level_values('label').values
    return train_df, train_labels, test_df, test_labels

def read_fold2(path, fold, dev_ratio=0, random_state=53, fold_count=-1):
    from sklearn.model_selection import train_test_split
    from pandas import DataFrame

    train_df, train_labels, test_df, test_labels = read_fold(path, fold)
    
    # consider a dev set?
    dev_df, dev_labels = DataFrame(), np.array([])
    if dev_ratio > 0: 
        # train=train_df.sample(frac=0.8,random_state=200)
        # test=df.drop(train.index)
        # [note] train_df has a multiple level index: (instance index, label)
        if dev_ratio < 1: 
            pass
        else: 
            div('dev_ratio is an integer greater than 1; will interpret it as the number of folds worth of data to take as development set.')
            if fold_count < 0: fold_count = 5
            assert dev_ratio < fold_count, "Cannot take more than {n} portions of the data as a dev set".format(n=dev_ratio)
            dev_ratio = dev_ratio / (fold_count + 0.0)
            print('... (verify) dev_ratio -> fold counts {fc}'.format(fc=dev_ratio))
        train_df, train_labels, dev_df, dev_labels = train_dev_split(train_df, ratio=dev_ratio, random_state=random_state)

    print('... (verify) type(dev_df): {0}, type(dev_labels): {1} => dim? {2} vs {3}'.format(type(dev_df), type(dev_labels), dev_df.shape, dev_labels.shape))
        
    return train_df, train_labels, dev_df, dev_labels, test_df, test_labels 
def train_dev_split(train_df, train_labels=None, ratio=0.1, random_state=None, index=-1):
    from sklearn.model_selection import train_test_split
    if random_state is None: random_state = int(time.time()+random.randint(1, 1000)+index)

    # train_set => train_set + dev_set
    labels = train_labels
    if labels is None: labels = train_df.index.get_level_values('label').values

    train_df, dev_df, train_labels, dev_labels = train_test_split(train_df, labels, test_size=ratio, 
            shuffle=True, stratify=labels, random_state=random_state)  # random_state

    ### alternative way of using train test split 
    #   but we need to use 'stratify' in order to sample classes proportional to their corresponding sample sizes
    # ids_train, ids_test = train_test_split(range(train_df.shape[0]), test_size=ratio, random_state=random_state)
    # trSet, devSet = train_df.iloc[ids_train], train_df.iloc[ids_test]
    # # [note] something this results in numpy.array
    # # train_df, train_labels, dev_df, dev_labels = train_test_split(train_df, train_labels, test_size=dev_ratio, random_state=random_state)
    # if train_labels is not None: 
    #     trL = train_labels[ids_train] 
    #     devL = train_labels[ids_test] 
    # else: 
    #     trL = trSet.index.get_level_values('label').values
    #     devL = devSet.index.get_level_values('label').values
    # return trSet, trL, devSet, devL
    return train_df, train_labels, dev_df, dev_labels

def subsample_array(A, axis=1, ratio=0.5, max_size=None, replace=False):
    """
    Input
    -----
    A: a numpy array
    axis: the dimension along which the observations (training instances) are indexed
          e.g. in user-by-item format, axis = 1, since 'items' correspond to training instances

    """ 
    # import numpy as np
    A = np.array(A)
    
    if A.ndim == 1:
        N = len(A)
    else:  
        N = A.shape[axis]
    
    if not max_size: 
        # use ratio
        max_size = int( np.ceil(N * ratio) )

    if max_size > N: max_size = N
    indices = np.random.choice(N, max_size, replace=replace)
    
    if A.ndim == 1: 
        return A[indices]
    elif A.ndim == 2: 
        if axis == 0: 
            return A[indices, :]
        else: 
            return A[:, indices]
    else: 
        # >= 3D: how to subset along a particular dimension? 
        raise NotImplementedError
    return A[:, indices]

def split(df, labels=[], ratio=0.2, shuffle=True, max_size=None, **kargs):
    """
    Split a dataframe into training and test splits in the format of 4-tuple: (train_df, test_df, train_labels, test_labels)


    Memo
    ----
    1. Create classification examples
       
       from sklearn import datasets
       X, y = datasets.make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=2, n_classes=2, weights=[0.8, 0.2])

    """
    import math
    from sklearn.model_selection import train_test_split
    
    # configure random state
    random_state = kargs.get('random_state', None)
    index = kargs.get('index', -1)
    if random_state is None: random_state = int(time.time()+random.randint(1, 1000)+index)
    labels = df.index.get_level_values('label').values

    if max_size and max_size < df.shape[0]: 
        train_size = int(math.ceil(max_size * ratio))
        test_size = max_size - train_size
        train_df, test_df, train_labels, test_labels = train_test_split(df, labels, train_size=train_size, test_size=test_size, shuffle=shuffle, stratify=labels, random_state=random_state)
        if max_size > 10: assert test_df.shape[0] == test_size
        
        ### alternatively, can use df.sample(n=, weights=) 
        #   however, we want to ensure that each class is sampled proportionally to its class-specific sample size

        # replace: False => sample without replacement
        # D = df.sample(n=max_size, random_state=kargs.get('random_state', 53), replace=kargs.get('replace', False))
        # if len(labels) > 0: 
        #     idx = dfp.index.get_level_values('id').values
        #     L = np.array(labels)[idx]
        # else: 
    else: 
        train_df, test_df, train_labels, test_labels = train_test_split(df, labels, test_size=ratio, shuffle=shuffle, stratify=labels, random_state=random_state)
        
    # output: train_df, test_df, train_labels, test_labels
    return train_df, test_df, train_labels, test_labels

def shuffle_split_cv(path, split_number=2, dev_ratio=-1, fold_count=-1, shuffle=False, max_size=None, random_state=None, fold=-1, verbose=True):
    """
    Read the CV fold of the base predictor-generated data (where the fold number is specified by 'fold') and return 
    either i) a train-test split, in which case, this routine is equivalent to read_fold() or 
           ii) a train-dev-test split

    Return type ii) is used to preserve the CV iteration on the test split while allowing for a further random subsampling on the training split 
    to produce a train-dev split; this is useful for performing model selection in the train-dev split. 
    """
    from sklearn.model_selection import train_test_split
    # import time

    # todo: max_size

    # import pandas as pd
    # import random
    # random.seed(random_state)  # [note] random has a different internal state than np.random
    # np.random.seed(random_state)  # cannot fix this; o.w. each function call to read_random_fold will generate the same sample
    if random_state is None: random_state = int(time.time()+random.randint(1, 1000)+fold)

    if fold_count < 0:  
        # need to infer fold count 
        file_type = 'validation'
        fold_count = 0
        readSuccess = True
        files_read = []
        while readSuccess:
            fpath = '{prefix}/{dtype}-{i}.csv.gz'.format(prefix=path, dtype=file_type, i=fold_count)   # e.g. wmf_F10_A100_Xbrier_CFitem_OPTrating_PTprior-validation-2.csv.gz
            if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
                fold_count += 1
                readSuccess = True
                files_read.append(fpath)
            else: 
                readSuccess = False 
                # fold_count -= 1   # the hypothesized fold_count does not exist, therefore, we backtrack by 1 (which exists)
        if not files_read: 
            fpath = '{prefix}/{dtype}*.csv.gz'.format(prefix=path, dtype=file_type)  # dtype: {'validation', 'prediction', 'prior', 'posterior', }
            if not glob.glob(fpath): 
                msg = "(read_random_fold) Error: Could not find any data matching %s" % fpath
                raise FileNotFoundError(msg)
        if verbose: print('(shuffle_split_cv) Inferred fold count: {n}'.format(n=fold_count))

    if dev_ratio < 0: dev_ratio = 1.0/fold_count  
    if dev_ratio > 0.0: 
        split_number_eff = 3
    else: 
        split_number_eff = 2
    print('(shuffle_split_cv) Fold {fo} | dev_ratio: {r}, split_number_eff: {sn} | path: {path}, exist? {tval}, size: {size}, fold count: {fc}'.format(fo=fold, 
        r=dev_ratio, sn=split_number_eff, path=path, tval=os.path.exists(path), size=os.path.getsize(path), fc=fold_count))

    # default value 
    train_df = dev_df = test_df = DataFrame()
    train_labels = dev_labels = test_labels = np.array([])

    train_df, train_labels, test_df, test_labels = read_fold(path, fold, shuffle=shuffle, fold_count=fold_count)
    print('... dim(train): {dtr}, dim(test): {dt}'.format(dtr=train_df.shape[0], dt=test_df.shape))
    if fold <= 1: 
        print('... label distribution: {ratio}'.format(ratio=collections.Counter(np.hstack([train_labels, test_labels]))))

    test_ratio = 1./fold_count
    if split_number_eff == 2: 
        # noop 
        pass
    else: 
        # otherwise, shuffle split training set to produce a train-dev split
      
        # ratio correction    
        dev_ratio = dev_ratio / (1.0-test_ratio)

        # NOTE: the return value from train_test_split() is different from the datasink convention!!!
        train_df, dev_df, train_labels, dev_labels = train_test_split(train_df, train_labels, test_size=dev_ratio, 
                shuffle=True, stratify=train_labels, random_state=random_state)

    if split_number == 2: 
        return train_df, train_labels, test_df, test_labels

    # special case: if split_number == 3 but dev_ratio == 0.0, then will still return 6-tuple but dev split will be empty
    return train_df, train_labels, dev_df, dev_labels, test_df, test_labels 

def shuffle_split(path, split_number=2, dev_ratio=0.2, test_ratio=0.2, fold_count=-1, max_size=None, random_state=None, index=-1):
    """
    Same as read_random_fold() but distinguishes the number of data splits controlled by 'split_number'

    Params
    ------

    Memo
    ----
    1. Also see sklearn.model_selection.ShuffleSplit
       but this routine does not seem to support 'stratify' according to the class proportion

    """
    if random_state is None: random_state = int(time.time()+random.randint(1, 1000)+index)

    if dev_ratio > 0.0: 
        split_number_eff = 3
    else: 
        split_number_eff = 2

    # todo: max_size
    train_df = dev_df = test_df = DataFrame()
    train_labels = dev_labels = test_labels = np.array([])
    if split_number_eff == 3: 
        train_df, train_labels, dev_df, dev_labels, test_df, test_labels = read_random_fold(path, 
            fold_count=fold_count, dev_ratio=dev_ratio, test_ratio=test_ratio, max_size=max_size, 
                random_state=random_state, shuffle=True, index=index, split_number=3)

        # return train_df, train_labels, dev_df, dev_labels, test_df, test_labels
    elif split_number_eff == 2: 
        train_df, train_labels, test_df, test_labels = read_random_fold(path, 
            fold_count=fold_count, test_ratio=test_ratio, max_size=max_size, 
                random_state=random_state, shuffle=True, index=index, split_number=2)

        # below would rely on the condition where dev_ratio > 0.0
        # train_df = concat([train_df, dev_df])
        # train_labels = np.hstack((train_labels, dev_labels))
        # return (train_df, train_labels, test_df, test_labels)
    else: 
        # msg = '(shuffle_split) Warning: Invalid split number: {n}'.format(n=split_number)
        # # print('(shuffle_split) Warning: Invalid split number: {n}'.format(n=split_number))
        # raise ValueError(msg)
        pass

    if split_number == 2: 
        return train_df, train_labels, test_df, test_labels

    return train_df, train_labels, dev_df, dev_labels, test_df, test_labels 
    
def read_random_fold(path, fold_count=-1, dev_ratio=-1, test_ratio=-1, max_size=None, random_state=None, shuffle=True, index=-1, split_number=3):
    """
    Read a random partition of the data from a CV fold. 

    Input
    -----


    Todo 
    ----
        max_size: maximum allowable sample size, default None => no upper limit

    Memo
    ----
    1. train_test_split

       random_state: 
           If int, random_state is the seed used by the random number generator; 
           If RandomState instance, random_state is the random number generator; 
           If None, the random number generator is the RandomState instance used by np.random
           default None
    """
    from sklearn.model_selection import train_test_split
    # import time

    # todo: max_size

    # import pandas as pd
    # import random
    # random.seed(random_state)  # [note] random has a different internal state than np.random
    # np.random.seed(random_state)  # cannot fix this; o.w. each function call to read_random_fold will generate the same sample
    if random_state is None: random_state = int(time.time()+random.randint(1, 1000)+index)

    if fold_count < 0:  
        # need to infer fold count 
        file_type = 'validation'
        fold_count = 0
        readSuccess = True
        files_read = []
        while readSuccess:
            fpath = '{prefix}/{dtype}-{i}.csv.gz'.format(prefix=path, dtype=file_type, i=fold_count)   # e.g. wmf_F10_A100_Xbrier_CFitem_OPTrating_PTprior-validation-2.csv.gz
            if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
                fold_count += 1
                readSuccess = True
                files_read.append(fpath)
            else: 
                readSuccess = False 
                # fold_count -= 1   # the hypothesized fold_count does not exist, therefore, we backtrack by 1 (which exists)
        if not files_read: 
            fpath = '{prefix}/{dtype}*.csv.gz'.format(prefix=path, dtype=file_type)  # dtype: {'validation', 'prediction', 'prior', 'posterior', }
            if not glob.glob(fpath): 
                msg = "(read_random_fold) Error: Could not find any data matching %s" % fpath
                raise FileNotFoundError(msg)

    fold = np.random.choice(range(fold_count), 1)[0]  # random.sample(range(fold_count), 1)[0]
    ###### 
    if shuffle: 
        # aggregate and then reshuffle
        dfs = []
        train_df = read_csv('%s/validation-%i.csv.gz' % (path, fold), index_col = [0, 1], compression = 'gzip')
        test_df = read_csv('%s/predictions-%i.csv.gz' % (path, fold), index_col = [0, 1], compression = 'gzip')
        train_size = train_df.shape[0]
        test_size = test_df.shape[0]
        dfs.extend([train_df, test_df])

        # aggregate 
        df = concat(dfs, axis = 0)

        # subsampling 
        if max_size and max_size < df.shape[0]: df = df.sample(n=max_size, replace=False)
        labels = df.index.get_level_values('label').values
        # print('... ids: {ids}'.format(ids=df.index.get_level_values('id').values[:100]))
        
        ### train-test split
        if test_ratio is None or test_ratio <= 0: test_ratio = 1/(fold_count+0.0)
        train_df, test_df, train_labels, test_labels = train_test_split(df, labels, test_size=test_ratio, 
            shuffle=True, stratify=labels, random_state=random_state)  # random_state
        assert all(train_df.index.get_level_values('label').values == train_labels)

        if split_number == 2:
            # NOTE: the return value from train_test_split() is different from the datasink convention!!! 

            return train_df, train_labels, test_df, test_labels
        
        elif split_number == 3: 
            # [test]
            # print('(read_random_fold) domain: %s | random fold: %d (/%d), size of data: %d | id ex: %s' % \
            #     (os.path.basename(path), fold, fold_count, train_df.shape[0]+test_df.shape[0], train_df.index.get_level_values('id').values[:10]))   # ... ok
            # assert all(test_df.index.get_level_values('label').values == test_labels)
            
            # df = df.sample(n=n_train+n_test)
            # print('... sampled ids: {ids}'.format(ids=df.index.get_level_values('id').values[:100]))
            
            ### train-dev split

            # adjust dev_ration according to remaining data after dropping the test portion
            dev_ratio = dev_ratio / (1.0-test_ratio)
            # dev_ratio = 1/(fold_count+0.0)
            
            train_df, dev_df, train_labels, dev_labels = train_test_split(train_df, train_labels, test_size=dev_ratio, 
                shuffle=True, stratify=train_labels, random_state=random_state)
        
            return train_df, train_labels, dev_df, dev_labels, test_df, test_labels

        # return train-test split by default 
        return train_df, test_df, train_labels, test_labels
    else:  
        print('(read_random_fold) domain: %s | random fold: %d, shuffle: False' % (os.path.basename(path), fold))
        # if dev_ratio == 0: 
        #     return read_fold(path, fold) 
        return read_fold2(path, fold, dev_ratio=dev_ratio, random_state=random_state)

def shuffle_split_reconstructed(path, method, split_number=2, dev_ratio=0.2, test_ratio=0.2, index=-1, max_size=None, 
        random_state=None, file_type='', fold_count=-1): 
    # two cases: 
    #     1. posterior data (the new training data D' after apply CF)
    #     2. iteration-ready data i.e. posterior data have been broken down into multiple train-test splits just like a CV dataset (dtype: 'validation' and 'prediction')
    if random_state is None: random_state = int(time.time()+random.randint(1, 1000)+index)

    if file_type.startswith(('pri', 'post')):  # prior: before transformation; 'posterior': after transformation 
        splits = read_reconstructed(path, method=method, max_size=max_size, 
            dev_ratio=dev_ratio, test_ratio=test_ratio, split_number=2, file_type=file_type, index=index, random_state=random_state)
        assert len(splits) == split_number * 2
        return splits  # (train_df, train_labels, test_df, test_labels) | (train_df, train_labels, dev_df, dev_labels, test_df, test_labels)
    else: 
        train_df, train_labels, dev_df, dev_labels, test_df, test_labels = read_random_fold_reconstructed(path, method=method, 
            fold_count=fold_count, dev_ratio=dev_ratio, test_ratio=test_ratio, max_size=max_size, random_state=random_state, index=index)
        if split_number == 3: 
            return (train_df, train_labels, dev_df, dev_labels, test_df, test_labels)
        elif split_number == 2: 
            train_df = concat([train_df, dev_df])
            train_labels = np.hstack((train_labels, dev_labels))
            return (train_df, train_labels, test_df, test_labels)
        else: 
            print('(shuffle_split_reconstructed) Warning: Invalid split number: {n}'.format(n=split_number))
            return (train_df, train_labels, dev_df, dev_labels, test_df, test_labels) 

def read_reconstructed(path, method, dev_ratio=None, test_ratio=0.2, file_type='posterior', max_size=None, split_number=2, index=-1, random_state=None):
    """
    Read the reconstructed data (file_type: posterior) and the original data (file_type: prior) for pairwise comparison (i.e. transformed vs original). 
    The data correspond to the test split of the CF-transfomred data and the test split of the original training data. 

    """

    from sklearn.model_selection import train_test_split
    if random_state is None: random_state = int(time.time()+random.randint(1, 1000)+index)

    # we need to break down the posterior data into proper train test splits, in this case 'fold' is irrelevant as each call will produce a different split 
    if index == -1: 
        fpath = '{prefix}/{id}-{dtype}.csv.gz'.format(prefix=path, id=method, dtype=file_type)   # e.g. wmf_F10_A100_Xbrier_CFitem_OPTrating_PTprior-posterior.csv.gz
    else: 
        fpath = '{prefix}/{id}-{dtype}-{index}.csv.gz'.format(prefix=path, id=method, dtype=file_type, index=index)   # e.g. wmf_F10_A100_Xbrier_CFitem_OPTrating_PTprior-posterior.csv.gz    

    assert os.path.exists(fpath), "Dataset (dtype: {dtype}, method ID: {id}) does not exist at {path}".format(dtype=file_type, id=method, path=fpath)
    df = read_csv(fpath, index_col=[0, 1], compression = 'gzip') 
    if index == 0: print('(read_reconstructed) Found posterior data (method ID: {id}) of dim: {dim} ... (verify) #'.format(id=method, dim=df.shape))

    # subsampling 
    if max_size and max_size < df.shape[0]: df = df.sample(n=max_size, replace=False)
    labels = df.index.get_level_values('label').values
    # print('... ids: {ids}'.format(ids=df.index.get_level_values('id').values[:100]))
    
    ### train-test split
    if test_ratio is None or test_ratio <= 0: test_ratio = 0.2
    train_df, test_df, train_labels, test_labels = train_test_split(df, labels, test_size=test_ratio, 
        shuffle=True, stratify=labels, random_state=random_state)  # random_state
    assert all(train_df.index.get_level_values('label').values == train_labels)
    
    ### train-dev split
    if split_number == 3 and dev_ratio is not None: 
        # if dev_ratio is None or dev_ratio <= 0: dev_ratio = 0.2
        train_df, dev_df, train_labels, dev_labels = train_test_split(train_df, train_labels, test_size=dev_ratio, 
            shuffle=True, stratify=train_labels, random_state=random_state)

        return (train_df, train_labels, dev_df, dev_labels, test_df, test_labels)
    
    return (train_df, train_labels, test_df, test_labels)

def read_random_fold_reconstructed(path, method, fold_count=-1, dev_ratio=0.2, test_ratio=0.2, max_size=None, random_state=None, index=-1): 
    from sklearn.model_selection import train_test_split
    import time

    # todo: max_size

    if random_state is None: random_state = int(time.time()+random.randint(1, 1000)+index)
    if fold_count < 0:  
        # need to infer fold count 
        fold_count = 0
        readSuccess = True
        files_read = []
        while readSuccess:
            fpath = '%s/%s-validation-%i.csv.gz' % (path, method, fold_count)
            # print('(read_random_fold_reconstructed) estimating fold count from %s' % fpath)
            
            if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
                fold_count += 1
                readSuccess = True
                # n_success += 1
                files_read.append(fpath)
            else: 
                readSuccess = False 
                # fold_count -= 1   # the hypothesized fold_count does not exist, therefore, we backtrack by 1 (which exists)
        if not files_read: 
            fpath = '%s/%s-validation*.csv.gz' % (path, method)
            if not glob.glob(fpath): 
                msg = "(read_random_fold_reconstructed) Error: Could not find any data matching %s" % fpath
                raise FileNotFoundError(msg)

    # print('(read_random_fold_reconstructed) fold_count: {fc}'.format(fc=fold_count))
    fold = np.random.choice(range(fold_count), 1)[0]  # random.sample(range(fold_count), 1)[0]
    train_df, train_labels, test_df, test_labels = read_fold_reconstructed(path, fold, method=method)

    # aggregate 
    df = concat([train_df, test_df], axis = 0)
    labels = df.index.get_level_values('label').values
    print('(read_random_fold_reconstructed) domain: %s | random fold: %d (/%d), size of data: %d' % (os.path.basename(path), fold, fold_count, df.shape[0]))
    # print('... ids: {ids}'.format(ids=df.index.get_level_values('id').values[:100]))
    
    ### train-test split
    if test_ratio is None or test_ratio <= 0: test_ratio = 1/(fold_count+0.0)
    train_df, test_df, train_labels, test_labels = train_test_split(df, labels, test_size=test_ratio, 
        shuffle=True, stratify=labels, random_state=random_state)  # random_state
    assert all(train_df.index.get_level_values('label').values == train_labels)

    ### train-dev split
    # if dev_ratio is None or dev_ratio <= 0: 
    dev_ratio = dev_ratio / (1.0-test_ratio)
    # dev_ratio = 1/(fold_count+0.0)
    train_df, dev_df, train_labels, dev_labels = train_test_split(train_df, train_labels, test_size=dev_ratio, 
        shuffle=True, stratify=train_labels, random_state=random_state)
    
    return train_df, train_labels, dev_df, dev_labels, test_df, test_labels
       
def read_subsampled(path, index, method, exception_=False, file_type='prior'):
    """

    Memo
    ----
    1. example training and test sets: 

    
    """

    ### load training data
    dtype = 'train-{ftype}'.format(ftype=file_type)
    fpath = '{prefix}/{id}-{dtype}-{index}.csv.gz'.format(prefix=path, id=method, dtype=dtype, index=index)   # e.g. wmf_F10_A100_Xbrier_CFitem_OPTrating_PTprior-posterior.csv.gz 
    try: 
        train_df        = read_csv(fpath, index_col=[0, 1], compression = 'gzip')  # index_col = [0, 1]
    except: 
        msg = "(read_subsampled) Subsample training split does not exist at (reconstructed? {tval}):\n{path}\n".format(tval=True if file_type.startswith('post') else False, path=fpath)
        # raise FileNotFoundError(msg)
        raise ValueError(msg)   

    ### load test data
    dtype = file_type
    fpath = '{prefix}/{id}-{dtype}-{index}.csv.gz'.format(prefix=path, id=method, dtype=dtype, index=index)   # e.g. wmf_F10_A100_Xbrier_CFitem_OPTrating_PTprior-posterior.csv.gz 
    try: 
        test_df        = read_csv(fpath, index_col=[0, 1], compression = 'gzip')  # index_col = [0, 1]
    except: 
        msg = "(read_subsampled) Subsampled test split does not exist at (reconstructed? {tval}):\n{path}\n".format(tval=True if file_type.startswith('post') else False, path=fpath)
        # raise FileNotFoundError(msg)
        raise ValueError(msg)   
    
    try: 
        train_labels    = train_df.index.get_level_values('label').values
        test_labels     = test_df.index.get_level_values('label').values
    except Exception as e: 
        msg = "Could not read dataset: {name}\n... {error}\n".format(name=os.path.basename(path), error=e)
        raise RuntimeError(msg)

    return train_df, train_labels, test_df, test_labels

def read_fold_reconstructed(path, fold, method, exception_=False, file_type='train-posterior'): 

    fpath = '%s/%s-validation-%i.csv.gz' % (path, method, fold) 
    try: 
        train_df        = read_csv(fpath, index_col=[0, 1], compression = 'gzip')  # index_col = [0, 1]
    except: 
        msg = "(read_fold_reconstructed) Reconstructed training set does not exist at:\n%s\n...('augmented' = False? )\n" % fpath
        # raise FileNotFoundError(msg)
        raise ValueError(msg)

    # read regular test set 
    reconstructed_testset = True
    if reconstructed_testset:
        # reconstructed testset may or may not exist: if we only reconstruct R but not T (when 'augmented' set to False), then reconstructed test set does not exist
        file_test_recons = '%s/%s-predictions-%i.csv.gz' % (path, method, fold)
        if os.path.exists(file_test_recons): 
            test_df =  read_csv('%s/%s-predictions-%i.csv.gz' % (path, method, fold), index_col = [0, 1], compression = 'gzip')
        else: 
            msg = "(read_fold_reconstructed) Reconstructed data does not exist at:\n%s\n...('augmented' = False? )\n" % file_test_recons
            if exception_: 
                raise ValueError(msg)
            else: 
                print(msg)
            ### substitue original test set 
            test_df = read_csv('%s/predictions-%i.csv.gz' % (path, fold), index_col = [0, 1], compression = 'gzip')
    else: 
        test_df = read_csv('%s/predictions-%i.csv.gz' % (path, fold), index_col = [0, 1], compression = 'gzip')

    try: 
        train_labels    = train_df.index.get_level_values('label').values
        test_labels     = test_df.index.get_level_values('label').values
    except Exception as e: 
        msg = "Could not read dataset: {name}\n... {error}\n".format(name=os.path.basename(path), error=e)
        raise RuntimeError(msg)

    return train_df, train_labels, test_df, test_labels

def reindex(path, method, file_type='validation', ext='csv.gz', exception_=True): 
    # import re, collections
    
    p_file = re.compile(r'(?P<method>\w+)\-%s-(?P<index>\d+)\.%s' % (file_type, ext))  # e.g. nmf_user_spectral_sim_F10-validation-0.csv.gz
    files, indices = [], []
    for f in os.listdir(path): # only list the basename NOT the full path
        m = p_file.match(f)
        if m: 
            method_id = m.group('method')
            if method == method_id:  # test if method (as a keyword) is part of the name
                print('(find_dataset_index) found a match > method: %s  ... (verify)' % m.group('method'))
                files.append(m.group('method'))
                indices.append(m.group('index'))

    counter = collections.Counter(files) # counts 
    files = set(files)  # unique names identified by the method_id
    N = 0
    if len(files) > 1:  # more than 1 set of data matching the input 'method'
        if exception_: 
            raise ValueError('Found more than one dataset that match the method:\n%s\n' % method)
        else: 
            N = max([v for (_, v) in counter.items()])
        # index = N - 1  # zero-based index
        # for i, (k, v) in enumerate(counter.items()): 
        #     if v != n_fold: 
        #         print('(match_exact) method %s may not have sufficient data, found only %d parts' % (k, v))
        #         files.remove(k)
    else: 
        the_method = files.pop()
        N = counter[the_method]

    print('(reindex) Re-indexing the dataset: {method_id} (0-{max_index}) ...'.format(method_id=method, max_index=N-1))
    # for i in range(N):
    #     fpath = os.path.join(path, )  

    return index

def match(path, method='', file_type='validation', ext='csv.gz', verify=True, exception_=False, keywords=[], 
            policy_iter='subsampling', dtype=int): # index_dtype=int
    """
    Match file names of the target datasets by method and a set of keywords. 
    The primary keyword 'method' is usually just a prefix representing the category of the algorithm that generated the dataset (e.g. wmf, nmf)
    
    cf: match_exact()
        method is expected to be very specific (e.g. 'nmf_item_sim_F10' in nmf_item_sim_F10-validation-1.csv.gz)

    Use 
    ---
    1. 'method' is usually a prefix of the dataset name (e.g. 'nmf' in nmf_item_sim_F10-validation-1.csv.gz)
    2. If the primary keyword 'method' is not specified, then we will proceed to use the keywords to do the match 
       If any one of the keywords finds it match in the dataset name, this dataset is considered a match (i.e. any, not all)

    """
    def add(adict, k, v, dtype=int):
        if not k in adict: adict[k] = []
        if hasattr(v, '__iter__'): 
            if dtype is not None: v = [dtype(e) for e in v]
            adict[k].extend(v)
        else: 
            if v is not None: 
                if dtype is not None: v = dtype(v)
                adict[k].append(v)
            else: 
                adict[k] = []  # no iteration index or CV fold count

    # import glob
    import re, collections
    # import numpy as np
    # e.g. nmf_item_sim_F10-validation-1.csv.gz
    
    # A. use globbing 
    # file_path = '{path}/{method}*-{dataset}-[0.csv.gz'.format(path=path, method=method, dataset=dtype, fold=fold)
    # for name in glob.glob('dir/*[0-9].*'):
    #     print(name) 

    # B. regex
    #    idiom: [f for f in os.listdir('.') if re.match(r'[0-9]+.*\.jpg', f)]
    if isinstance(keywords, str): keywords = [keywords, ]

    files = {}
    p_file = re.compile(r'(?P<method>\w+)\-%s-(?P<index>\d+)\.%s' % (file_type, ext))  # e.g. nmf_user_spectral_sim_F10-validation-0.csv.gz
    for f in os.listdir(path): 
        m = p_file.match(f)
        if m: 
            # print('(match) method: %s' % m.group('method'))
            method_id = m.group('method')
            index = m.group('index')
            if not method or method in method_id:  # if method (as a primary keyword) is not specified (then we'll rely on the other keywords) or method is part of the file name (e.g. 'nmf' in nmf_item_sim_F10-validation-1.csv.gz)
                
                # other keywords? 
                if len(keywords) > 0:  
                    matched_bits = np.zeros(len(keywords))
                    for i, keyword in enumerate(keywords): 
                        if keyword in method_specific: 
                            matched_bits[i] = 1
                    # all() or any()? 
                    if any(matched_bits): 
                        add(files, method_id, index, dtype=int)
                else: 
                    add(files, method_id, index, dtype=int)
    
    ##############################################
    # ... Given the files multibag 

    if verify: 
        if policy_iter.startswith(('cv', 'cross')): 
            if len(files) > 0: 
                n_fold = max([len(v) for (_, v) in files.items()])
                for i, (k, v) in enumerate(files.items()): 
                    if len(v) != n_fold: 
                        print('(match) method %s may not have sufficient data, found only %d parts' % (k, v))
                        files.pop(k)
                print('... number of CV folds: %d' % n_fold)
            else: 
                print('... No data found with method: %s' % method)
        else: 
            pass  # do nothing
        
    else: 
        # there may be more than one set that matches but hopefully not. [assumption] the input 'method' (really a method ID) should be as specific as possible
        # >>> see utils_cf.MFEnsemble.get_tset_id(method=method, params=params, meta_params=meta_params)
        if len(files) > 1: 
            msg = "(match_exact) Found more than one matching dataset (n={n}). Ambiguous method ID: {mid}\n".format(n=len(files), mid=method)
            msg += "~>\n{d}\n".format(d=files)
            if exception_: 
                raise ValueError(msg)
            # else: 
            #     if verify: print(msg)
    return files

def match_exact_pair(path, method, file_type='prior', ext='csv.gz', verify=True, exception_=False, policy_iter='subsampling'):
    """

    Memo
    ----
    1. file_type: 'prior' must pair with 'train-prior'
        'prior' in this case correspond to the test set
        'train-prior' corresponds to the training set

    """
    def add(adict, k, v, dtype=int):
        if not k in adict: adict[k] = []
        if hasattr(v, '__iter__'): 
            if dtype is not None: v = [dtype(e) for e in v]
            adict[k].extend(v)
        else: 
            if v is not None: 
                if dtype is not None: v = dtype(v)
                adict[k].append(v)
            else: 
                adict[k] = []  # particular iteration index or CV fold count
    def intersection(first, *others):
        # find intersection of n-way sets
        return set(first).intersection(*others)

    import re, collections, sys

    methods = method
    if isinstance(method, str):
        methods = [method, ]   
    print('... methods: {0}'.format(methods)) 

    files = {}  # 'files' are really just the 'stems' of the file name minus the suffix 'validation-<CV number>'.csv.gz
    file_types = {'train': 'train-{dtype}'.format(dtype=file_type), 'test': file_type, }  # training and test split

    for method in methods: 
        print('### processing method: {m}'.format(m=method))
        # indices = []  # indices may not be contiguous in the case of random subsampling 
        matched_ids = set()
        for split_type, file_type in file_types.items():
            if not split_type in files: files[split_type] = {} 

            # p_file = re.compile(r'(?P<method>\w+)\-%s-(?P<index>\d+)\.%s' % (file_type, ext))  # e.g. nmf_user_spectral_sim_F10-validation-0.csv.gz
            p_file = re.compile(r'(?P<method>\w+)\-%s(-(?P<index>\d+))?\.%s' % (file_type, ext))  # index is optional
            
            for f in os.listdir(path): # foreach filename under the path (not full paths)
                m = p_file.match(f)
                if m: 
                    method_id = m.group('method')
                    if method_id is not None: 
                        if method == method_id:  # test if method (as a keyword) is part of the name
                            add(files[split_type], k=method_id, v=m.group('index'), dtype=int) # keep track of the indices because in random subsampling mode, these indices may not be contiguous as in CV
                            matched_ids.add(method_id)
                            print('   + (match_exact_pair) found a match > method: {mid}, fold/id: {id} | dtype: {dtype} ... (verify)'.format(mid=method_id, id=m.group('index'), dtype=file_type))
                    else: # bp 
                        add(files[split_type], k=method, v=m.group('index'), dtype=int)
                        print('   + (match_exact_pair) found a match from base predictors (BP) fold/id: {id} | dtype: {dtype} ... (verify)'.format(mid=method, id=m.group('index'), dtype=file_type))
            print('-' * 80)

    print('(debug) files: {val}\n'.format(val=files))

    methods_base = {}
    files_paired = {}
    for split_type in ['train', 'test', ]:
        entry = files[split_type]
        print('... {type}-split: '.format(type=split_type))
        if split_type.startswith('tr'):
            for method_id, indices in entry.items(): 
                print('...... method_id: {id} => {alist}'.format(id=method_id, alist=indices))
                methods_base[method_id] = 0
                # indices_base.extend(indices)   # both training and test split needs to share the index set, so that they can serve as input to a classifier
                files_paired[method_id] = indices

        else: # other splits such as the test split
            for method_id, indices in entry.items(): 
                if method_id in files_paired:  
                    files_paired[method_id] = sorted(set(files_paired[method_id]).intersection(indices))
                    methods_base[method_id] += 1
                print('...... found pair-able data > dataset ID: {method_id}, indices: {ids}'.format(method_id=method_id, ids=files_paired[method_id]))

    # finally remove the entry with training data but without test data 
    for method_id, state in methods_base.items(): 
        if state == 0:  # never matched in the test (or other splits)
            files_paired.pop(method_id)
            print("!!! method ID: {id} exist in the training split but not non-training splits !!!".format(id=method_id))

    print('(match_exact_pair) Found a total of {n} paired datasets'.format(n=len(files_paired)))

    return files_paired 

def match_exact(path, method, file_type='validation', ext='csv.gz', verify=True, exception_=False, policy_iter='subsampling', mode='train-test-split'):
    def add(adict, k, v, dtype=int):
        if not k in adict: adict[k] = []
        if hasattr(v, '__iter__'): 
            if dtype is not None: v = [dtype(e) for e in v]
            adict[k].extend(v)
        else: 
            if v is not None: 
                if dtype is not None: v = dtype(v)
                adict[k].append(v)
            else: 
                adict[k] = []  # particular iteration index or CV fold count

    # import glob
    import re, collections
    # e.g. nmf_item_sim_F10-validation-1.csv.gz

    # assert isinstance(method, (str, list, set, )

    if mode.startswith('train'):
        # method can be a list 
        return match_exact_pair(path, method, file_type='prior', ext='csv.gz', verify=True, exception_=False, policy_iter='subsampling') 
    
    ############################################################
    # mode == 'pair'
    methods = method
    if isinstance(method, str):
        methods = [method, ] 
    else: 
        methods = method
        assert isinstance(method, (list, set, ))
        print('(match_exact) multiple method IDs: {0}'.format(methods))

    files = {}  # 'files' are really just the 'stems' of the file name minus the suffix 'validation-<CV number>'.csv.gz
    # indices = []  # indices may not be contiguous in the case of random subsampling 

    for method in methods: 
        if method.startswith(('bp', 'base')):
            p_file = re.compile(r'((?P<method>\w+)\-)?%s-(?P<index>\d+)\.%s' % (file_type, ext)) 
        else:   
            # p_file = re.compile(r'(?P<method>\w+)\-%s-(?P<index>\d+)\.%s' % (file_type, ext))  # e.g. nmf_user_spectral_sim_F10-validation-0.csv.gz
            p_file = re.compile(r'(?P<method>\w+)\-%s(-(?P<index>\d+))?\.%s' % (file_type, ext))  # index is optional
        
        for f in os.listdir(path): # foreach filename under the path (not full paths)
            m = p_file.match(f)
            if m: 
                method_id = m.group('method')
                if method_id is not None: 
                    if method == method_id:  # test if method (as a keyword) is part of the name
                        add(files, k=method_id, v=m.group('index'), dtype=int) # keep track of the indices because in random subsampling mode, these indices may not be contiguous as in CV
                        print('   + (match_exact) found a match > method: {mid}, fold/id: {id} | dtype: {dtype} ... (verify)'.format(mid=method_id, id=m.group('index'), dtype=file_type))
                else: # bp 
                    add(files, k=method, v=m.group('index'), dtype=int)
                    print('   + (match_exact) found a match from base predictors (BP) fold/id: {id} | dtype: {dtype} ... (verify)'.format(mid=method, id=m.group('index'), dtype=file_type))

    if verify: 
        if policy_iter.startswith(('cv', 'cross')): 
            if len(files) > 0: 
                n_fold = max([len(v) for (_, v) in files.items()])
                for i, (k, v) in enumerate(files.items()): 
                    if len(v) != n_fold: 
                        print('(match_exact) method %s may not have sufficient data, found only %d parts' % (k, v))
                        files.pop(k)
                print('... number of CV folds: %d' % n_fold)
            else: 
                print('... No data found with method: %s' % method)
        else: 
            pass  # do nothing
        
    else: 
        # there may be more than one set that matches but hopefully not. [assumption] the input 'method' (really a method ID) should be as specific as possible
        # >>> see utils_cf.MFEnsemble.get_tset_id(method=method, params=params, meta_params=meta_params)
        if len(files) > 1: 
            msg = "(match_exact) Found more than one matching dataset (n={n}). Ambiguous method ID: {mid}\n".format(n=len(files), mid=method)
            msg += "~>\n{d}\n".format(d=files)
            if exception_: 
                raise ValueError(msg)
            # else: 
            #     if verify: print(msg)
                # sorted( [(key, len(bag)) for key, bag in files.items()], key=lambda x: x[1], reverse=True)[0] 
    return files  # a dictionary that maps method to indices (typically there's only one method)

import numpy as np

def weighted_median(data, weights):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights

    Reference
    ---------
    1. https://gist.github.com/tinybike/d9ff1dad515b66cc0d87#file-weighted_median-py

    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    return w_median

def test_weighted_median():
    data = [
        [7, 1, 2, 4, 10],
        [7, 1, 2, 4, 10],
        [7, 1, 2, 4, 10, 15],
        [1, 2, 4, 7, 10, 15],
        [0, 10, 20, 30],
        [1, 2, 3, 4, 5],
        [30, 40, 50, 60, 35],
        [2, 0.6, 1.3, 0.3, 0.3, 1.7, 0.7, 1.7, 0.4],
    ]
    weights = [
        [1, 1/3, 1/3, 1/3, 1],
        [1, 1, 1, 1, 1],
        [1, 1/3, 1/3, 1/3, 1, 1],
        [1/3, 1/3, 1/3, 1, 1, 1],
        [30, 191, 9, 0],
        [10, 1, 1, 1, 9],
        [1, 3, 5, 4, 2],
        [2, 2, 0, 1, 2, 2, 1, 6, 0],
    ]
    answers = [7, 4, 8.5, 8.5, 10, 2.5, 50, 1.7]
    for datum, weight, answer in zip(data, weights, answers):
        assert(weighted_median(datum, weight) == answer)

def rmse_score(a, b):
    return sqrt(mean((a - b)**2))

def unbag0(df, bag_count):
    cols = []
    bag_start_indices = range(0, df.shape[1], bag_count)
    names = [col.split('.')[0] for col in df.columns.values[bag_start_indices]]
    # print('... (verify) classifier names: {names} <- {orig}'.format(names=names, orig=df.columns.values))
    for i in bag_start_indices:
        cols.append(df.iloc[:, i:i+bag_count].mean(axis = 1)) # average all bags
    df = concat(cols, axis = 1)
    df.columns = names
    return df

def infer_bag_count(names, sep='.', verify=False):
    from collections import Counter
    counts = Counter([name.split(sep)[0] for name in names])

    unames = list(counts.keys())
    bag_count = counts[unames[0]]  # assuming that all classifiers have the same bag count

    if verify:
        # assuming that we do not mixed unbagged with bagged
        for name, count in counts.items(): 
            if count != bag_count: 
                msg = "Inconsistent bag counts: %s" % counts
                if exception_: 
                    raise ValueError(msg)
                else: 
                    print(msg)
        print('(infer_bag_count) inferred bag_count=%d' % bag_count)
    return (unames, bag_count)

def unbag(df, bag_count=None, sep='.', exception_=False):
    # import re
    import pandas as pd
    from collections import Counter

    # first ensure that columns are sorted according to their names and bag indices 
    df = df.reindex(sorted(df.columns), axis=1) # df[sorted(df.columns.values)]

    # is this a bagged dataframe? 
    # p_col = re.compile(r'^(?P<bp>\w+)\.(?P<bag>\d+)$')

    # n_bagged_cls = sum([1 for col in df.columns.values if p_col.match(col)]) 
    # print('(test) columns:\n%s\n' % df.columns.values)
    n_bagged_cls = sum([1 for col in df.columns.values if len(str(col).split(sep))==2] )  # raise exception when col is not a strong (e.g. numbers)
    tBagged = True if n_bagged_cls > 0 else False
    if not tBagged: 
        msg = "(evaluate.unbag) Input dataframe does not contain bagged models:\n%s\n" % df.columns.values
        if exception_: 
            raise ValueError(msg)
        else: 
            print(msg)

        # nothing else to do 
        return df

    # infer bag count if None
    if tBagged and bag_count is None: 
        counts = Counter([col.split(sep)[0] for col in df.columns.values])
        bag_count = counts[list(counts.keys())[0]]

        # assuming that we do not mixed unbagged with bagged
        for name, count in counts.items(): 
            if count != bag_count: 
                msg = "Inconsistent bag counts: %s" % counts
                if exception_: 
                    raise ValueError(msg)
                else: 
                    print(msg)
        print('(unbag) inferred bag_count=%d' % bag_count)

    cols = []
    bag_start_indices = range(0, df.shape[1], bag_count)
    names = [_.split(sep)[0] for _ in df.columns.values[bag_start_indices]]
    for i in bag_start_indices:
        cols.append(df.iloc[:, i:i+bag_count].mean(axis = 1))
    df = pd.concat(cols, axis = 1)
    df.columns = names
    return df 

diversity_score = average_pearson_score
score = sklearn.metrics.roc_auc_score
greater_is_better = True
best = max if greater_is_better else min
argbest = argmax if greater_is_better else argmin
fmax_scorer = sklearn.metrics.make_scorer(fmax_score, greater_is_better = True, needs_threshold = True)


def demo_sampling(): 
    from analyze_performance import Analysis
    from stacking import read
    import collections
    import getpass
    from numpy import linalg as LA

    # debugging 
    np.set_printoptions(precision=3)

    domain = 'pf2' # 'diabetes_cf'
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path

    dev_ratio = 0.0
    test_ratio = 0.6
    # setting dev_ration to 0 => ValueError: The test_size = 0 should be greater or equal to the number of classes = 2

    train_df, train_labels, dev_df, dev_labels, test_df, test_labels = shuffle_split(project_path, split_number=3, dev_ratio=dev_ratio, test_ratio=test_ratio, 
        fold_count=-1, max_size=None, random_state=None)

    N = train_df.shape[0] + dev_df.shape[0] + test_df.shape[0]
    print('> sizes | n(train): {ntr}, n(dev): {ndev}, n(test): {nt} | N: {N}'.format(ntr=train_df.shape[0], ndev=dev_df.shape[0], nt=test_df.shape[0], N=N))
    print('> types | t(train): {ttr}, t(dev): {tdev}, t(test): {tt}'.format(ttr=type(train_df), tdev=type(dev_df), tt=type(test_df)  ))

    idx = train_df.index.get_level_values('id').values
    print('> train_id: {idx}'.format(idx=idx[:10]))
    idx_test = test_df.index.get_level_values('id').values
    print('> test_id: {idx}'.format(idx=idx_test[:10]))

    return

def t_match(**kargs):
    from analyze_performance import Analysis
    from stacking import read
    import collections

    domain = 'pf1'
    Analysis.config(domain=domain)
    project_path = Analysis.project_path
    analysis_path = Analysis.analysis_path
    ##########################################
    # ... project path and analysis path defined

    import getpass
    user = getpass.getuser() # 'pleiades' 
    # domain = 'pf3'
    # project_path = '/Users/%s/Documents/work/data/%s' % (user, domain)  # /Users/pleiades/Documents/work/data/recommender

    methods = match(path=project_path, method='', file_type='validation', ext='csv.gz', verify=True) 
    print(methods)
    # ... example: 
    #     {'nmf': [0, 1, 2, 3, 4], 'nmf_sim_user_sim_F10': [0], 'wmf': [0, 1, 2, 3, 4], 'wmf_F100_A100_Xbrier_CFuser_OPTrating_PTprior_UnSpv': [0, 1, 2, 3, 4], ... }

    # exact math 
    # e.g. wmf_F1_A1_Xbrier_CFitem_OPTrating_PTprior-posterior.csv.gz
    # method_id = 'wmf_F100_A100_Xbrier_CFuser_OPTrating_PTprior_S8' # 'wmf_F1_A1_Xbrier_CFitem_OPTrating_PTprior'
    dset_id = 'wmf_F75_A100_XCFuser_S2'
    dtype = 'posterior'
    # datasets = match_exact(path=project_path, method=dset_id, file_type=dtype, ext='csv.gz', verify=True, policy_iter='subsampling')
    # datasets = match_exact(path=project_path, method=dset_id, file_type='posterior', ext='csv.gz', verify=True, exception_=False, policy_iter='subsampling')
    # print(datasets)
    # ... exmaple: 
    #     {'wmf_F1_A1_Xbrier_CFitem_OPTrating_PTprior': []}  
    #     note that 'prior' or 'posterior' data do not have fold count or index
    case = 1
   
    if case == 0: 
        # proxy test for stacking module 
        for method_id, indices in datasets.items(): 
            # for fold in range(7): 
            #     train_df, train_labels, test_df, test_labels = read(fold, path=project_path, dataset=method_id, policy_iter='subsampling', file_type='posterior')

            #     idx = test_df.index.get_level_values('id').values[:10]
            #     print('... dim(train_df): {dim_tr}, dim(test_df): {dim_t}  | index: {idx}'.format(dim_tr=train_df.shape, dim_t=test_df.shape, idx=idx))

            #     train_counts = collections.Counter(train_labels) 
            #     test_counts = collections.Counter(test_labels)
            #     print('... counts(train): {ctr} | counts(test): {ct} | ratio(train): {rtr}, ratio(test): {rt}'.format(ctr=train_counts, 
            #         ct=test_counts, rtr=train_counts[1]/sum(train_counts.values()), rt=test_counts[1]/sum(test_counts.values()) ))
            
            for train_df, train_labels, test_df, test_labels in readAllIter(path=project_path, dataset=method_id, file_type='posterior', n_runs=7):
                print('... dim(tset): {dim}, labels: {l}'.format(dim=test_df.shape, l=test_labels[:10]))

                idx = test_df.index.get_level_values('id').values[:10]
                print('... dim(train_df): {dim_tr}, dim(test_df): {dim_t}  | index: {idx}'.format(dim_tr=train_df.shape, dim_t=test_df.shape, idx=idx))

                train_counts = collections.Counter(train_labels) 
                test_counts = collections.Counter(test_labels)
                print('... counts(train): {ctr} | counts(test): {ct} | ratio(train): {rtr} ~? ratio(test): {rt}'.format(ctr=train_counts, 
                    ct=test_counts, rtr=train_counts[1]/sum(train_counts.values()), rt=test_counts[1]/sum(test_counts.values()) ))
    elif case == 1: 
        mode_evaluation = kargs.get('mode', 'train-test-split')
        mode_evaluation = 'pair'
        file_types = ['prior', 'posterior', ]

        dset_id = 'wmf_F75_A100_XCFuser_S2'

        # but sometimes, we need to allow for a mixture of different models (when each cycle of model selection concludes a different set of 'best params')
        dset_ids = ['wmf_F75_A100_XCFuser_S2', 'wmf_F100_A100_XCFuser_S2', ]

        for file_type in file_types: 
            datasets = match_exact(path=project_path, method=dset_ids, file_type=file_type, ext='csv.gz', verify=True, policy_iter='subsampling', mode=mode_evaluation)
            print('> File type: {ftype} | Found {n} datasets:\n{adict}\n'.format(ftype=file_type, n=len(datasets), adict=datasets))
            
            for dataID, indices in datasets.items():
                n_sets = 0
                for index in indices: 
                    train_df, train_labels, test_df, test_labels = read(index, 
                            path=project_path, 
                            dataset=dataID, 
                            file_type=file_type,  # values: {'prior', 'posterior', 'train' }
                            policy_iter='subsampling',   # values: {'cv', 'subsampling'}
                            mode=mode_evaluation) # common.read_fold(project_path, fold)
                    n_sets +=1
                print('... file type: {ftype} | read {n} sets of data'.format(ftype=file_type, n=n_sets)) 


    return

def t_utility(**kargs): 

    # infer bag count
    names = ['a.0', 'a.1', 'a.2', 'b.0', 'b.1', 'b.2']
    unames, n_bags = infer_bag_count(names, sep='.', verify=False)
    print("(t_utility) names: {}, n_bags: {}".format(unames, n_bags))

def t_distance(): 

    ld = levenshtein
    print(
    ld('kitten','kitten'), # 0
    ld('kitten','sitten'), # 1
    ld('kitten','sittes'), # 2
    ld('kitten','sityteng'), # 3
    ld('kitten','sittYing'), # 4
    ld('rosettacode','raisethysword'), # 8 
    ld('kitten','kittenaaaaaaaaaaaaaaaaa'), # 17
    ld('kittenaaaaaaaaaaaaaaaaa','kitten') # 17
)

def test(**kargs):

    ### matching datasets by names (e.g. method_id generated by MFEnsemble.get_dset_id())
    # demo_match()

    # demo_sampling()

    ### utilty functions 
    # demo_utility()

    ### distance metric 
    # demo_distance()

    ### matrix operations 
    X = np.random.randint(0, 5, (10, 10))
    Xp = perturb(X, lower_bound=0, alpha=100.)
    print("[test] X:\n{}\n".format(X))
    print("...    Xp:\n{}\n".format(Xp))

    return 

if __name__ == "__main__": 
    test()
