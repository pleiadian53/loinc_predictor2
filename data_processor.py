import os, sys

import sklearn.datasets as datasets

from pandas import DataFrame
import pandas as pd
from tabulate import tabulate
import numpy as np

dataDir = os.path.join(os.getcwd(), 'data')  # default unless specified otherwise

class Data(object): 
    label = 'test_result_loinc_code'
    features = []
    prefix = os.path.join(os.getcwd(), 'data')

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
    canonicalized = kargs.get('canonicalized', False)

    if not input_file: 
        if cohort: 
            input_file = f"ts-{cohort}.csv" 
        else: 
            input_file = "ts-generic.csv"
    input_path = os.path.join(input_dir, input_file)

    df = pd.read_csv(input_path, sep=sep, header=0, index_col=None, error_bad_lines=False, warn_bad_lines=warn_bad_lines)
    if verbose: print("(load_data) Loaded dataframe (dim={}) from:\n{}\n".format(df.shape, input_path))
    
    if canonicalized: 
        import loinc as lc
        col_target = kargs.get('col_target', 'test_result_loinc_code')
        token_default = token_missing = 'unknown'
        df = df.drop_duplicates(keep='last')  # drop duplicates 
        df = lc.canonicalize(df, col_target=col_target, token_missing=token_default) # noisy_values/[]

    if len(columns) > 0: 
        return df[columns]

    return df

def generate_data(case='classification', sparse=False):
    """
    Example for generating training instances.
    """
    from sklearn import datasets
    from sklearn.utils import shuffle
    from scipy.sparse.csr import csr_matrix

    """Generate regression/classification data."""
    bunch = None
    if case == 'regression':
        bunch = datasets.load_boston()
    elif case == 'classification':
        bunch = datasets.fetch_20newsgroups_vectorized(subset='all')
    X, y = shuffle(bunch.data, bunch.target)
    offset = int(X.shape[0] * 0.8)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    if sparse:
        X_train = csr_matrix(X_train)
        X_test = csr_matrix(X_test)
    else:
        X_train = np.array(X_train)
        X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_train = np.array(y_train)
    data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train,
            'y_test': y_test}
    return data

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

def balance_data_incr(df, df_extern, n_samples=-1, col='test_result_loinc_code', labels=[], verbose=1, **kargs):
    """
    Similar to balance_data() but expect df_extern to be huge (to a degree that may not fit into main memory)
    """
    import loinc as ul

    n_baseline = n_samples
    if n_baseline < 0: 
        ds = stratify(df, col=col)
        # use the median sample size as the standard sample size
        n_baseline = int(np.median([e[1] for e in ds]))
    print("(balance_data_incr) n_baseline={}".format(n_baseline))

    # labels: all the possible class labels (e.g. LOINC codes) read from 'col' (e.g. test_result_loinc_code)
    if len(labels) == 0: 
        labels = set(df[col])

        # verify the external data
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

def group_by(df, cols=['test_result_code', 'test_result_name',], verbose=1, n_samples=-1): 
    """
    Group training data by patient attributes. 

    Note that there is a groupby counterpart for the LOINC table (see loinc.py)

    """
    cols_key = ['test_result_loinc_code', ]
    cols_test = ['test_order_code', 'test_order_name', 'test_result_code', 'test_result_name', ]
    cols_text = ['medivo_test_result_type', 'test_result_comments', ] 
    target_properties = cols_key + cols_test + cols_text

    adict = {}
    for index, dfg in df.groupby(cols): 
        dfe = dfg[target_properties]
        adict[index] = dfe

    if verbose:
        n_groups = len(adict)
        if n_samples < 0: n_samples = n_groups

        test_cases = set(np.random.choice(range(n_groups), min(n_groups, n_samples)))
        for i, (index, dfg) in enumerate(adict.items()):
            nrow = dfg.shape[0] 
            if i in test_cases and nrow > 1: 
                print("... [{}] => \n{}\n".format(index, dfg.sample(n=min(nrow, 5)).to_string(index=False) ))
    
    return adict

def toXY(df, cols_x=[], cols_y=[], untracked=[], **kargs): 
    
    verbose = kargs.get('verbose', 1)
    
    X = y = None
    if len(untracked) > 0: 
        df = df.drop(untracked, axis=1)

    if len(cols_x) > 0: 
        X = df[cols_x].values
        
        cols_y = list(df.drop(cols_x, axis=1).columns)
        y = df[cols_y].values

    else: 
        assert len(cols_y) > 0 
        cols_x = list(df.drop(cols_y, axis=1).columns)
        X = df[cols_x].values
        y = df[cols_y].values

    # optional operations
    scaler = kargs.get('scaler', None)
    if scaler is not None: 
        if verbose: print("(toXY) Scaling X using method={}".format(scaler))
        from sklearn import preprocessing
        if scaler.startswith('standard'): 
            std_scale = preprocessing.StandardScaler().fit(X)
            X = std_scale.transform(X)
        elif scaler.startswith(('normalize', 'minmax')): 
            minmax_scale = preprocessing.MinMaxScaler().fit(X)
            X = minmax_scale.transform(X)
    
    # return (X, y, z, cols_x, cols_y)
    return (X, y, cols_x, cols_y)

def down_sample(df, col_label='label', n_samples=-1):
    labelCounts = dict(df[col_label].value_counts())
    labelSorted = sorted(labelCounts, key=labelCounts.__getitem__, reverse=True)
    max_label, min_label = labelSorted[0], labelSorted[-1]
    max_count, min_count = labelCounts[max_label], labelCounts[min_label]

    df_max = df.loc[df[col_label]==max_label]
    df_min = df.loc[df[col_label]==min_label]

    n_samples = min(n_samples, min_count) if n_samples > 0 else min_count

    # downsample majority classs
    df_max = df_max.sample(n=n_samples)

    if n_samples < df_min.shape[0]: 
        df_min = df_min.sample(n=n_samples)
    return pd.concat([df_min, df_max], ignore_index=True)

def map_group_to_label(df, cols=['test_result_code', 'test_result_name',], verbose=1, n_samples=-1):
    return

########################################################################################################

def get_diabetes_data(): 
    # https://www.kaggle.com/uciml/pima-indians-diabetes-database#diabetes.csv
    
    fpath = 'data/diabetes.csv'
    # diabetes dataset
    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    # load dataset  
    pima = pd.read_csv(fpath, header=1, names=col_names)  # header=None
    print("> columns: {}".format(col_names))
    
    feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose', 'bp', 'pedigree']
    X = pima[feature_cols] # Features
    y = pima.label # Target variable
    
    print("> data layout:\n{}\n".format(pima.head()))
    
    return (X, y, feature_cols) 

def load_data0(input_path=None, input_file=None, col_target='label', exclude_vars=[],  sep=',', verbose=True): 
    """

    Memo
    ----
    1. input file examples: 
        a. multivariate imputation applied 
             exposures-4yrs-merged-imputed.csv
        b. rows with 'nan' dropped 
             exposures-4yrs-merged.csv
    """
    import collections

    col_label = Data.label
    
    if input_path is None: 
        assert input_file is not None 
        input_path = os.path.join(dataDir, input_file)
    else: 
        if os.path.isdir(input_path): 
            assert input_file is not None 
            input_path = os.path.join(input_path, input_file)
        else: 
            # if verbose: print("(load_data) input_path is a full path to the dataset: {}".format(input_path))
            pass
        
    assert os.path.isfile(input_path), "Invalid input path: {}".format(input_path)

    df = pd.read_csv(input_path, header=0, sep=sep) 
    
    exclude_vars = list(filter(lambda x: x in df.columns, exclude_vars))
    exclude_vars.append(col_target)
    dfx = df.drop(exclude_vars, axis=1)
    features = dfx.columns.values
    
    X = dfx.values
    y = df[col_target]

    if verbose: 
        counts = collections.Counter(y)
        print("... class distribution | classes: {} | sizes: {}".format(list(counts.keys()), counts))
        print("... dim(X): {} variables: {}".format(X.shape, features))
    
    return (X, y, features)

def load_merge(vars_matrix='exposures-4yrs.csv', label_matrix='nasal_biomarker_asthma1019.csv', 
        output_matrix='exposures-4yrs-asthma.csv',
        backup=False, save=True, verify=False, sep=',', imputation=False):
    """
    Load source data (e.g. asthma biomarkers) and merge them with another labeled dataset (e.g. pollutant matrix). 


    Memo
    ----
    1. IterativeImputer: 

       A strategy for imputing missing values by modeling each feature with missing values as a function of other features in a round-robin fashion.

    """
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from shutil import copyfile
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
 
    null_threshold = 0.8
    generic_key = 'ID'
    tImputation = imputation

    # A. load variables 
    primary_key = 'Study.ID'
    droplist_vars = ['DoB' , 'Zipcode', ]  # 'Zipcode',  'Gender'
    # ... DoB needs further processing 
    categorical_vars = ['Zipcode',  'Gender']

    fpath_vars = os.path.join(dataDir, vars_matrix)
    assert os.path.exists(fpath_vars), "(load_merge) Invalid path to the variable matrix"
    dfv = pd.read_csv(fpath_vars, header=0, sep=sep) 
    print("(load_merge) dim(dfv): {} | vars:\n... {}\n".format(dfv.shape, dfv.columns.values))

    # backup 
    # if backup: 
    #     vars_matrix_bk = "{}.bk".format(vars_matrix)
    #     fpath_vars_bk = os.path.join(dataDir, vars_matrix_bk)
    #     copyfile(fpath_vars, fpath_vars_bk)
    #     print("... copied {} to {} as a backup".format(vars_matrix, vars_matrix_bk))

    # transform
    msg = ''
    le = LabelEncoder()
    dfvc = dfv[categorical_vars]
    msg += "... original dfvc:\n"
    msg += tabulate(dfvc.head(5), headers='keys', tablefmt='psql') + '\n'

    dfvc2 = dfvc.apply(le.fit_transform)
    msg += "... transformed dfvc:\n"
    msg += tabulate(dfvc2.head(5), headers='keys', tablefmt='psql')
    dfv[categorical_vars] = dfvc2[categorical_vars]

    # mapping
    if verify: 
        for cvar in categorical_vars:
            mapping = dict(zip(dfvc[cvar], dfvc2[cvar]))
            cols = [cvar, '{}_numeric'.format(cvar)]
            dfmap = DataFrame(mapping, columns=cols)
            msg += tabulate(dfmap.head(10), headers='keys', tablefmt='psql') + '\n'

            uvals = dfvc[cvar].unique()
            for uval in uvals: 
                print("... {} | {} -> {}".format(cvar, uval, mapping[uval]))

    # drop columns 
    dfv = dfv.drop(droplist_vars, axis=1)    

    # rename ID column 
    dfv = dfv.rename(columns={primary_key: generic_key})
    msg += "... full transformed data:\n"
    msg += tabulate(dfv.head(3), headers='keys', tablefmt='psql') + '\n'

    msg += "... final dim(vars_matrix): {} | vars:\n... {}\n".format(dfv.shape, dfv.columns.values)
    N0 = dfv.shape[0]
    assert generic_key in dfv.columns
    print(msg)
    ####################################################

    # B. load labels (from yet another file)
    primary_key = 'Study ID'

    label = 'Has a doctor ever diagnosed you with asthma?' 
    labelmap = {label: 'label'}

    # only take these columns
    col_names = [primary_key, label, ]
    categorical_vars = [label, ]
    # ... use the simpler new variable names?
    
    fpath_labels = os.path.join(dataDir, label_matrix)
    assert os.path.exists(fpath_labels), "(load_merge) Invalid path to the label matrix"
    dfl = pd.read_csv(fpath_labels, header=0, sep=sep)
    dfl = dfl[col_names]
    assert dfl.shape[1] == len(col_names)
    print(tabulate(dfl.head(5), headers='keys', tablefmt='psql'))
    
    # rename
    # dfl = dfl.rename(columns={primary_key: generic_key, label: labelmap[label]})
    # print("(load_merge) dim(dfl): {} | cols:\n... {}\n".format(dfl.shape, dfl.columns.values))
    
    # transform
    msg = ''
    le = LabelEncoder()
    dflc = dfl[categorical_vars]
    msg += "... original dflc:\n"
    msg += tabulate(dflc.head(5), headers='keys', tablefmt='psql') + '\n'
    print(msg)

    # dflc2 = dflc.apply(le.fit_transform)  # log: '<' not supported between instances of 'str' and 'float'

    # transform column by column 
    cols_encoded = {cvar: {} for cvar in categorical_vars}
    for cvar in categorical_vars: 
        cvar_encoded = '{}_numeric'.format(cvar)
        dfl[cvar] = le.fit_transform(dfl[cvar].astype(str))

        # keep track of mapping
        cols_encoded[cvar] = {label:index for index, label in enumerate(le.classes_)}  # encoded value -> name 

    # dfl[categorical_vars] = dflc2[categorical_vars]

    # mapping
    if verify: 
        msg += "> verifying the (value) encoding ...\n"
        for cvar in categorical_vars:
            mapping = cols_encoded[cvar]
            for label, index in mapping.items(): 
                print("... {} | {} -> {}".format(cvar, label, index))    

    # rename
    colmap = {primary_key: generic_key, label: labelmap[label]}
    msg += "> renaming columns via {}\n".format(colmap)
    dfl = dfl.rename(columns=colmap)
    msg += "... transformed dflc:\n"
    msg += tabulate(dfl.head(5), headers='keys', tablefmt='psql') + '\n'

    # merge
    msg += "> merging variable matrix and label matrix ...\n"
    dfv = pd.merge(dfv, dfl, on=generic_key, how='inner')
    ###################################################

    # finally drop ID columns so that only explanatory and response variables remain
    dfv = dfv.drop([generic_key, ], axis=1)

    msg += "... final dim(vars_matrix) after including labels: {} | vars:\n... {}\n".format(dfv.shape, dfv.columns.values)
    assert dfv.shape[0] == N0, "Some IDs do not have data? prior to join n={}, after join n={}".format(N0, dfv.shape[0]) 

    example_cols = ['Gender', 'ECy1', 'SO4y1', 'NH4y1', 'Nity1', 'OCy1', 'label', ]
    msg += tabulate(dfv[example_cols].head(10), headers='keys', tablefmt='psql') + '\n'

    # remove rows with all NaN? 
    # null_threshold = 0.8
    msg += "... dropping rows with predominently null values (thresh={})\n".format(null_threshold)
    Nu, Ni = dfv.shape 

    # if tDropNA: 
    Nth = int(Ni * null_threshold)  # Keep only the rows with at least Nth non-NA values.
    msg += "...... keeping only rows with >= {} non-NA values\n".format(Nth)
    # dfv.dropna(thresh=Nth, axis=0, inplace=True) # how='all',
    dfv.dropna(how='any', inplace=True) 
    msg += "> after dropping rows with predominently null, n_rows: {} -> {} | dim(dfv): {}\n".format(N0, dfv.shape[0], dfv.shape)

    if tImputation:
        msg += "... applying multivarite data imputation\n"
        imp = IterativeImputer(max_iter=60, random_state=0)
        dfx = dfv.drop(['label', ], axis=1)
        features = dfx.columns
        X = dfx.values
        y = dfv['label']

        X = imp.fit_transform(X)
        dfv = DataFrame(X, columns=features)
        dfv['label'] = y
        assert dfv.shape[0] == dfv.dropna(how='any').shape[0], "dfv.shape[0]: {}, dfv.dropna.shape[0]: {}".format(dfv.shape[0], dfv.dropna(how='any').shape[0])

    msg += "... data after dropping NAs and/or imputation:\n"
    example_cols = ['Gender', 'ECy1', 'SO4y1', 'NH4y1', 'Nity1', 'OCy1', 'label', ]
    msg += tabulate(dfv[example_cols].head(10), headers='keys', tablefmt='psql') + '\n'

    print(msg)

    if save: 
        output_path = os.path.join(dataDir, output_matrix)
        dfv.to_csv(output_path, sep=sep, index=False, header=True)
        print('(load_merge) Saved output dataframe to: {}'.format(output_path))

    return dfv

def test(**kargs): 

    load_merge(vars_matrix='exposures-4yrs.csv', 
                 label_matrix='nasal_biomarker_asthma1019.csv', 
                    output_matrix='exposures-4yrs-asthma.csv')

    return

if __name__ == "__main__": 
    test()

