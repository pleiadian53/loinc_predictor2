import statistics 
import numpy as np 
import random, math, re
import pandas as pd
from pandas import Series, DataFrame
from bisect import bisect

import warnings 
warnings.filterwarnings("ignore")
#################################
# e.g. 
#     opt/anaconda3/lib/python3.6/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: 
#     Function safe_indexing is deprecated; safe_indexing is deprecated in version 0.22 and will be removed in version 0.24.
#     warnings.warn(msg, category=FutureWarning)

def weighted_choice(choices):
    """
    choices: 
       [("WHITE",90), ("RED",8), ("GREEN",2)]
    """
    values, weights = zip(*choices)
    total = 0
    cum_weights = []
    for w in weights:
        total += w
        cum_weights.append(total)
    x = random.random() * total
    i = bisect(cum_weights, x)
    return values[i]

def bootstrap_resample(X, y=None, n=None, all_labels_present=True, n_cycles=20):
    """ 
    Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    import collections
    if isinstance(X, pd.Series):
        X = X.copy()
        X.index = range(len(X.index))
    else: 
         X = np.array(X) # need to use array/list to index elements
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int) # e.g. array([ 8410, 11437, 87128, ..., 75103,  5866, 44852])
    X_resample = np.array(X[resample_i])  # TODO: write a test demonstrating why array() is important

    if y is not None:
        labels = np.unique(y)
        n_labels = len(labels)
        assert len(y) == len(X)

        y_resample = np.array(y[resample_i]) 
        if all_labels_present: 
            n_labels_resample = len(np.unique(y_resample))
            while True:
                if n_labels_resample == n_labels: break 
                # need to resample again 
                if j > n_cycles: 
                    print('bootstrap> after %d cycles of resampling, still could not have all labels present.')
                    ac = collections.Counter(y)
                    print('info> class label counts:\n%s\n' % ac) 
                    break

                resample_i = np.floor(np.random.rand(n)*len(X)).astype(int) # e.g. array([ 8410, 11437, 87128, ..., 75103,  5866, 44852])
                X_resample = np.array(X[resample_i])  # TODO: write a test demonstrating why array() is important
                y_resample = np.array(y[resample_i]) 

                j += 1 
            # assert np.unique(y_resample) == n_labels

        return (X_resample, y_resample)
    return X_resample

def bootstrap(population, k):
    return sample_wr(population, k) 
def sample_wr0(population, k):
    return [random.choice(population) for _ in range(k)]
def sample_wr(population, k):
    "Chooses k random elements (with replacement) from a population"
    n = len(population)
    _random, _int = random.random, int  # speed hack 
    result = [None] * k
    for i in range(k):
        j = _int(_random() * n)
        result[i] = population[j]
    return result
# --- alias ---
sample_with_replacement = sample_wr

def ci(scores, low=0.05, high=0.95):
    sorted_scores = np.array(scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(low * len(sorted_scores))]
    confidence_upper = sorted_scores[int(high * len(sorted_scores))]
    # print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
    #     confidence_lower, confidence_upper)) 
    return (confidence_lower, confidence_upper)

def ci2(scores, low=0.05, high=0.95, mean=None):
    std = statistics.stdev(scores) 
    mean_score = np.mean(scores)  # bootstrap sample mean
    if mean is None: mean = mean_score

    sorted_scores = np.array(scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(low * len(sorted_scores))]
    confidence_upper = sorted_scores[int(high * len(sorted_scores))]
    middle = (confidence_upper+confidence_lower)/2.0  # assume symmetric

    print('ci2> mean score: %f, middle: %f' % (mean_score, middle))
    # mean = sorted_scores[int(0.5 * len(sorted_scores))]
    # print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
    #     confidence_lower, confidence_upper)) 

    if confidence_upper > 1.0: 
        print('ci2> Warning: upper bound larger than 1.0! %f' % confidence_upper)
        confidence_upper = 1.0

    # this estimate may exceeds 1 
    delminus, delplus = (mean-confidence_lower, confidence_upper-mean)

    return (confidence_lower, confidence_upper, delminus, delplus, std)

def ci3(scores, low=0.05, high=0.95):
    if isinstance(scores[0], int): 
        scores = [float(e) for e in scores]
    sorted_scores = np.array(scores)
    sorted_scores.sort()
    mean_score = np.mean(scores)  # bootstrap sample mean
    se = statistics.stdev(scores) # square root of sample variance, standard error

    confidence_lower = sorted_scores[int(low * len(sorted_scores))]
    confidence_upper = sorted_scores[int(high * len(sorted_scores))]
    # print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
    #     confidence_lower, confidence_upper)) 
    return (mean_score, se, confidence_lower, confidence_upper)

def ci4(scores, low=0.05, high=0.95):
    if isinstance(scores[0], int): 
        scores = [float(e) for e in scores]

    ret = {}
    sorted_scores = np.array(scores)
    sorted_scores.sort()
    ret['mean'] = np.mean(scores)  # bootstrap sample mean
    ret['median'] = np.median(scores)
    ret['se'] = ret['error'] = se = statistics.stdev(scores) # square root of sample variance, standard error

    ret['ci_low'] = ret['confidence_lower'] = confidence_lower = sorted_scores[int(low * len(sorted_scores))]
    ret['ci_high'] = ret['confidence_upper'] = confidence_upper = sorted_scores[int(high * len(sorted_scores))]
    # print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
    #     confidence_lower, confidence_upper)) 

    return ret # keys: ci_low, ci_high, mean, median, se/error
# --- alias --- 
compute_scoring_stats = ci4

def sorted_interval_sampling(l, npar, reverse=False):
    """
    Arguments
    ---------
    npar: n partitions 
    """ 
    l.sort(reverse=reverse)
    avg = len(l)/float(npar)
    slist, partitions = [], []
    last = 0.0

    while last < len(l):
        partitions.append(l[int(last):int(last + avg)])
        last += avg    
    
    npar_eff = len(partitions) # sometimes 1 extra 
    # print('info> n_par: %d' % len(partitions))
    # print('\n%s\n' % partitions)
    for par in partitions:
        slist.append(random.sample(par, 1)[0])
        
    # 0, 1, 2, 3, 4, 5 => n=6, 6/2=3, 6/2-1=2 
    # 0, 1, 2, 3, 4    => n=5, 5/2=2 
    if npar_eff > npar: 
        assert npar_eff - npar == 1
        del slist[npar_eff/2]

    assert len(slist) == npar
    # for par in [l[i:i+n] for i in xrange(0, len(l), n)]: 
    #     alist.append(random.sample(par, 1)[0])
    return slist

# sampling with datastruct 
def sample_dict(adict, n_sample=10): 
    """
    Get a sampled subset of the dictionary. 
    """
    import random 
    keys = adict.keys() 
    n = len(keys)
    keys = random.sample(keys, min(n_sample, n))
    return {k: adict[k] for k in keys} 

def sample_subset(x, n_sample=10):
    if len(x) == 0: return x
    if isinstance(x, dict): return sample_dict(x, n_sample=n_sample)
    
    # assume [(), (), ] 
    return random.sample(x, n_sample)

def sample_cluster(cluster, n_sample=10): 
    """
    Input
    -----
    cluster: a list of cluster indices 
             e.g. 3 clusters 7 data points [0, 1, 1, 2, 2, 0, 0] 

    """
    n_clusters = len(set(cluster))
    hashtb = {cid: [] for cid in cluster}

    for i, cid in enumerate(cluster): 
        hashtb[cid].append(i)      # cid to positions        
 
    return sample_hashtable(hashtb, n_sample=n_sample)
# --- alias ---
sample_from_cluster = sample_cluster

def partitions(n):
    # base case of recursion: zero is the sum of the empty list
    if n == 0:
        yield []
        return
        
    # modify partitions of n-1 to form partitions of n
    for p in partitions(n-1):
        print('+ yield [1] + %s' % p)
        yield [1] + p
        if p and (len(p) < 2 or p[1] > p[0]):
            print('+ yield %s + %s' % ([p[0] + 1], p[1:]))
            yield [p[0] + 1] + p[1:]

# def sampleTSet(X, y, **kargs):

#     if kargs.has_key('n_per_class'): 
#         y_pos = X.shape[1]
#         df = DataFrame(np.hstack((X, y)))
#         sampleDataframe()


#     return 

def splitData(y, ratios=[0.8, 0.1, 0.1], min_r=1e-5, random_state=None):
    """
    Given label
    """
    from pandas import Series 
    from math import floor
    from itertools import cycle

    rT = sum(ratios)
    assert rT <= 1.0, "invalid proportions (must be <= 1.0): %s" % ratios
    if 1.0 - rT > min_r: 
        ratios.append(1.0-rT)

    if random_state is not None: 
        # this does not fix the permutation
        np.random.seed(random_state)

    N, nR = len(y), len(ratios)
    permuted_idx = np.random.permutation(N)  # permute all elements in y
    y_indexed = y
    if isinstance(y, Series):
        permuted_idx = np.random.permutation(y.index) # permute the given indices (e.g. a subset of multilabel data)
    else:  
        y_indexed = Series(y)
      
    # partition size(y) ~ ratios  
    splits = []
    n_samples = N
    for r in ratios: 
        if n_samples > 0: 
            split = int(floor(N*r))
            splits.append(split)
            n_samples -= split
        else: 
            print('Warning: No data points left for the split (r=%f)' % r)
            splits.append(0)
    assert n_samples >= 0
    if n_samples > 0:
        for ith in cycle(range(nR)):
            if n_samples <= 0: break
            splits[ith] += 1 
            n_samples -= 1

    assert sum(splits) == N
    
    # print('checks> N=%d => splits: %s' % (N, splits))
    
    # from splits to cut indices 
    cuts = [0, ]
    for i, incr in enumerate(splits): 
        cuts.append(incr+cuts[i])
    assert cuts[-1] == N

    ridx = []
    for i in range(nR): 
        idx = permuted_idx[cuts[i]:cuts[i+1]]
        # y_parts.append(y_indexed.iloc[idx].values)
        ridx.append(idx)

    return ridx # randomly subsampled positions of y ~ ratios

def splitDataPerClass(y, ratios=[0.8, 0.1, 0.1, ], min_r=1e-5, random_state=None):
    yp = Series(y)
    labels = yp.unique()

    # ratios must some to 1 
    rT = sum(ratios)
    assert rT <= 1.0, "invalid proportions (must be <= 1.0): %s" % ratios
    if 1.0 - rT > min_r: 
        ratios.append(1.0-rT)
    nR = len(ratios)

    ridx_label = [[] for _ in range(nR)]
    for label in labels: # foreach label ... 
        yl = yp.loc[yp == label]
        # print('  + label: %s | n=%d' % (label, len(yl)))
        ridx = splitData(yl, ratios, random_state=random_state) # ... split data following same ratios

        # assert sum(len(rids) for rids in ridx) == len(yl)
        for i, rids in enumerate(ridx):  # foreach ratio 
            ridx_label[i].extend(rids)
    assert sum(len(rids) for rids in ridx_label) == len(yp)

    return ridx_label  # len(ridx_label) == len(ratios)

############################################################
# --- Sampling from dataframe 

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

def sampleDataframe(df, col=None, **kargs):
    """
    
    Params
    ------
    **kargs
      n
      frac
      replace 
      weights

    Memo
    ----
    a. subsetting a datatframe according to a condition
        c_code = df['icd9']==code
        c_id = df['mrn']==mrn
        rows = df.loc[c_code & c_id] 

    """
    if col is None:  
        return df.sample(**kargs)  # params: n, frac, replace, weights 

    ### sample according to the strata defined by 'col'
    # assert col in df.columns
    
    # A. ensure each stratum gets at most N samples
    if kargs.has_key('n_per_class'): 
        strata = df[col].unique()
        n_per_class = kargs['n_per_class']
        print('sampleDataframe> n_classes=%d, n_per_class=%d' % (len(strata), n_per_class))
        dfx = []
        for i, stratum in enumerate(strata): 
            dfp = df.loc[df[col]==stratum]
            n0 = dfp.shape[0]  # n_per_class cannot exceed n0
            dfx.append(dfp.sample(n=min(n_per_class, n0)))
        df = pd.concat(dfx, ignore_index=True)
        assert df.shape[0] <= len(strata) * n_per_class
        return df 
    else: 
        raise NotImplementedError("sampleDataframe> Unknown sampling mode.")
    
    return df

############################################################

def sample_hashtable(hashtable, n_sample=10):
    import random, gc, copy
    from itertools import cycle

    n_sampled = 0
    tb = copy.deepcopy(hashtable)
    R = list(tb.keys()); random.shuffle(R) # shuffle elements in R inplace 
    nT = sum([len(v) for v in tb.values()])
    print('sample_hashtable> Total keys: %d, members: %d' % (len(R), nT))
    
    n_cases = n_sample 
    candidates = set()

    for e in cycle(R):
        if n_sampled >= n_cases or len(candidates) >= nT: break 
        entry = tb[e]
        if entry: 
            v = random.sample(entry, 1)
            candidates.update(v)
            entry.remove(v[0])
            n_sampled += 1

    return candidates 

def random_items(iterable, k=1):
    result = [None] * k
    for i, item in enumerate(iterable):
        if i < k:
            result[i] = item
        else:
            j = int(random.random() * (i+1))
            if j < k:
                result[j] = item
    random.shuffle(result)
    return result

def random_items2(iterable, k=1):
    from heapq import nlargest
    return (x for _, x in nlargest(k, ((random.random(), x) for x in iterable)))

def random_items3(iterable, samplesize):
    results = []
    iterator = iter(iterable)
        
    # Fill in the first samplesize elements:
    try:
        for _ in xrange(samplesize):
            results.append(iterator.next())
    except StopIteration:
        # raise ValueError("Sample size larger than population.")
        print("Warning: sample size %d larger than population > use only %d instances" % (samplesize, len(results)))
        assert len(results) > samplesize/2, "Insufficient pairs, not getting even half > n=%d" % len(results)

    random.shuffle(results)  # Randomize their positions
    for i, v in enumerate(iterator, samplesize):
        r = random.randint(0, i)
        if r < samplesize:
            results[r] = v  # at a decreasing rate, replace random items
    return results

def demo_resample(): 
    # X = np.array(range(10000)) # arange(10000)
    from sklearn import datasets
    from sklearn.linear_model import LogisticRegression
    from sklearn.grid_search import GridSearchCV
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import roc_curve, auc

    mu, sigma = 1, 2 # mean and standard deviation
    X = np.random.normal(mu, sigma, 1000)
    nX = len(X)
    X_resample = bootstrap_resample(X, n=5000)
    nXs = len(X_resample)
    print('size of X: %d, Xs: %d' % (nX, nXs))
    print('X: %s' % X_resample[:10])
    print('original mean:', X.mean())
    print('resampled mean:', X_resample.mean())

    reg = np.logspace(-3, 3, 7)
    penalty = 'l1'
    clf0 = LogisticRegression(tol=0.01, penalty='l1', solver='saga')
    params = [{'C': reg }] 
    estimator = GridSearchCV(clf0, params, cv=10, scoring='roc_auc')

    print('> test resampling training examples')
    X, y = datasets.make_classification(n_samples=10000, n_features=20,
                                    n_informative=2, n_redundant=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    estimator.fit(X_train, y_train)
    probas_ = estimator.predict_proba(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    auc_score = auc(fpr, tpr)

    print('> prior to resample dim X_train: %s, dim y_train: %s => auc: %f' % (str(X_train.shape), str(y_train.shape), auc_score))

    scores = []
    Xr, yr = X_train[:], y_train[:]
    for i in range(30): 
        assert Xr.shape == X_train.shape and len(y_train) == len(yr)
        estimator = GridSearchCV(clf0, params, cv=10, scoring='roc_auc')
        estimator.fit(Xr, yr)
        probas_ = estimator.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        auc_score = auc(fpr, tpr)
        print('  + resampled => auc: %f' %  auc_score)
        scores.append(auc_score)
        Xr, yr = bootstrap_resample(Xr, yr)

    print('  + mean auc: %f, median auc: %f' % (np.mean(scores), np.median(scores)))

    return X_resample 

def test_bsr_mean():
    # test that means are close
    np.random.seed(123456)  # set seed so that randomness does not lead to failed test
    X = arange(10000)
    X_resample = bootstrap_resample(X, n=5000)
    assert abs(X_resample.mean() - X.mean()) / X.mean() < 1e-2, 'means should be approximately equal'

def demo_weighted_sampling():
    x = [("WHITE",90), ("RED",8), ("GREEN",2)] 
    for i in range(100): 
        o = weighted_choice(x)
        print(o)

def demo_interval_sampling(): 
    alist = list(range(0, 20))
    elements = sorted_interval_sampling(alist, 3)
    print('> sampled(n=%d):\n%s\n' % (len(elements), elements))

    return 

def demo_cluster(): 
    cluster = [0, 1, 1, 2, 2, 0, 0, 2, 1]
    candidate_indices = sample_cluster(cluster, n_sample=4)
    candidates = ['_'] * len(cluster)
    for i in candidate_indices: 
        candidates[i] = 'x'

    cstr = ' '.join(str(e) for e in cluster)
    sstr = ' '.join(str(e) for e in candidates)

    print(cstr) 
    print(sstr)

    return 

def test_partition(): 
    
    print(list(partitions(4)))

    return

def demo_subsampling(**kargs):
    from sklearn.datasets import make_moons, make_circles, make_classification

    X, y = make_classification(n_samples=100, n_features=10, n_classes=5, n_redundant=1, n_informative=5,
                           random_state=1, n_clusters_per_class=3)

    print("> number of classes: %d | size(X): %d" % (len(np.unique(y)), len(X)))

    ratios = [0.3, ] # [0.8, 0.1, 0.1]

    ridx = splitData(y, ratios=ratios)
    ngs = [len(rids) for rids in ridx]
    nEst = sum(len(rids) for rids in ridx)
    print('check> n_groups: %d | %s' % (len(ridx), ngs)); assert nEst == len(X), "sum(len(rids) for rids in ridx)=%d" % nEst
    
    # for i, rids in enumerate(ridx): 
    #     print('  + %s' % rids)
    #     if i > 0: 
    #        assert len(set(ridx[i]).intersection(ridx[i-1])) == 0

    ridx = splitDataPerClass(y, ratios=ratios)  # training, dev, test
    for i, rids in enumerate(ridx): 
        print('  + %s' % rids)
        if i > 0: 
           assert len(set(ridx[i]).intersection(ridx[i-1])) == 0
        print('n_classes (ratio/split #%d): %d'  % (i+1, len(np.unique(y[rids]))))

    return

def demo_ci(scores=[]):
    import collections 
    # m, se, cil, cih = ci3(scores)
    # print('mean: %f, stderr: %f, CI:(%f, %f)' % (m, se, cil, cih))

    scores = [0.92, 0.75, 0.78, 0.89, 0.99, 0.98, ]
    ret = ci4(scores, low=0.05, high=0.95)  # keys: ci_low, ci_high, mean, median, se/error
    print('... mean: %f, median: %f, ci_low:%f | ci_high:%f ' % (ret['mean'], ret['median'], ret['ci_low'], ret['ci_high']))

    scores2 = bootstrap_resample(scores, n=100)
    print('... original:\n%s\n' % scores)
    print('... resampled:\n%s\n' % scores2)

    ret = ci4(scores2, low=0.05, high=0.95)  # keys: ci_low, ci_high, mean, median, se/error
    print('... mean: %f, median: %f, ci_low:%f | ci_high:%f ' % (ret['mean'], ret['median'], ret['ci_low'], ret['ci_high']))

    return

def demo_iter(): 
    n = 100
    pairs = ((i, j) for i in range(n) for j in range(i+1, n))
    pairs = random_items2(iterable=pairs, k=10)
    
    n_pairs = len(list(pairs)) # sum(1 for x in pairs)
    print('> n_pairs: %d' % n_pairs)

    for i, pair in enumerate(pairs): 
        print("[%d] %s" % (i, pair))

    return

def test(): 

    ### Confidence interval estimate
    demo_ci()

    demo_weighted_sampling()
    demo_interval_sampling()

    ### resampling 
    Xs = demo_resample()
    demo_ci(Xs)

    # sample cluster
    demo_cluster()

    ### some neat algorithms useful for sampling processes
    test_partition()
    demo_subsampling()

    # samping a generator and an iterator 
    demo_iter()

    return

if __name__ == "__main__": 
    test()