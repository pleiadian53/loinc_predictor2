import os, sys

import sklearn.datasets as datasets

from pandas import DataFrame
import pandas as pd
from tabulate import tabulate

dataDir = os.path.join(os.getcwd(), 'data')  # default unless specified otherwise

class Data(object): 
    label = 'label'
    features = []

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

def load_data(input_path=None, input_file=None, col_target='label', exclude_vars=[],  sep=',', verbose=True): 
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

def encode_vars(df):
    """
    Encode categorical and ordinal features. 

    References
    ----------
    1. dealing with categorical data

       a. https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63

       b. one-hot encoding

          https://www.ritchieng.com/machinelearning-one-hot-encoding/ 

    2. One-hot encoding in sklearn 

        a. The input to this transformer should be a matrix of integers, denoting the values taken on by categorical (discrete) features.
        b. The output will be a sparse matrix where each column corresponds to one possible value of one feature.
        c. It is assumed that input features take on values in the range [0, n_values).
        d. This encoding is needed for feeding categorical data to many scikit-learn estimators, notably linear models and SVMs with the standard kernels.
    """
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    # le = LabelEncoder()

    # # first we need to know which columns are ordinal, categorical 
    # col = ''
    # num_labels = le.fit_transform(df[col])
    # mappings = {index: label for index, label in enumerate(le.classes_)} 

    # limit to categorical data using df.select_dtypes()
    df = df.select_dtypes(include=[object])
    print(df.head(3))
    
    cols = df.columns  # categorical data candidates  <<< edit here

    le = preprocessing.LabelEncoder()

    # 2/3. FIT AND TRANSFORM
    # use df.apply() to apply le.fit_transform to all columns
    df2 = df.apply(df.fit_transform)
    print(df2.head())
    # ... now all the categorical variables have numerical values

    # INSTANTIATE
    enc = preprocessing.OneHotEncoder()

    # FIT
    enc.fit(df2)

    # 3. Transform
    onehotlabels = enc.transform(df2).toarray()
    

    return

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

