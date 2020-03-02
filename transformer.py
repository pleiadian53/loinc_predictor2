import category_encoders as ce
import numpy as np
import pandas as pd

def encode_vars_via_lookup(fset, feature_lookup): 
    # import category_encoders as ce
    
    for var in fset: 

        encoder = None
        vtype = feature_lookup.get(var, 'numeric') # default numeric
        
        if vtype == 'ord': 
            encoder = ce.OrdinalEncoder(cols=[var, ])
        elif vtype == 'cat': 
            encoder = ce.OneHotEncoder(cols=[var, ])
        elif vtype in ('str', 'high_card'): # high_card: categorical but with high cardinality
            encoder = ce.BinaryEncoder(cols=[var, ])  # ... or use ce.HashingEncoder()
        else: 
            # assuming that the var is numeric
            pass 

        # data imputation [todo]

        if encoder is not None: 
            dfX = encoder.fit_transform(dfX, dfy)
    return dfX
            
def encode_vars0(dfX, fset, high_card_cols=[], dfy=None, verbose=1): 
    # import category_encoders as ce
    dfX, _ = encode_vars(dfX, fset, high_card_cols=high_card_cols, dfy=dfy, verbose=verbose)
    return dfX

def encode_vars(dfX, fset, high_card_cols=[], dfy=None, verbose=1): 
    # import category_encoders as ce

    low_card_cols = list(set(fset)-set(high_card_cols))
    if verbose: print("(encoder_vars2) low card vars (n={}):\n{}\n ... high card vars (n={}):\n{}\n".format(
        low_card_cols, len(low_card_cols), high_card_cols, len(high_card_cols)))
    
    n_trans = 0
    for var in fset: 

        encoder = None
        if var in low_card_cols: 
            encoder = ce.OneHotEncoder(cols=[var, ])
        elif var in high_card_cols: # categorical but with high cardinality
            encoder = ce.BinaryEncoder(cols=[var, ])  
            # ... or use ce.HashingEncoder()

        # data imputation

        if encoder is not None: 
            n_trans += 1
            print(f'... transforming var: {var} ...')
            if dfy is not None: 
                dfX = encoder.fit_transform(dfX, dfy)
            else: 
                dfX = encoder.fit_transform(dfX)
    assert n_trans > 0

    return dfX, encoder

def categorify(df, cat_cols, cont_cols=[],  verbose=1): 
    for cat in cat_cols:
        if cat in df.columns: 
            df[cat] = df[cat].astype('category')

    print("(categorify) dtypes: {}".format(df.dtypes))

    # if not cont_cols: 
    #     cont_cols = list(set(df.columns)-set(cat_cols))

    # # inspect the code
    # if verbose: 
    #     for col in cols:
    #         print(f"> [{col}] =>\n  {df[col].head().cat.codes}\n")

    #     X_cats = np.stack([df[col].cat.codes.values for col in cat_cols], 1)
    #     X_conts = np.stack([df[col].values for col in cont_cols], 1)

    #     nf = X_cats.shape[1] + X_conts.shape[1]
    #     print("> Using n={} features ... #".format(nf))

    #     # to (X, y)
    #     X = np.hstack([X_cats, X_conts])
    #     # y = df[target_cols[0]].values
    #     # print('> dim(X): {}, dim(y): {}'.format(X.shape, y.shape))
    return df

def remove_null_like_values(source_values, extended='%'): 
    import string 

    punctuations = string.punctuation + extended

    new_values = []
    for source_value in source_values: 
        if pd.isna(source_value): continue
        new_value = str(source_value).translate(str.maketrans('', '', punctuations)).strip()
        if len(new_value) > 0: 
            new_values.append(new_value)
        # if removing punctuations result in empty string, then the value may carry no information (e.g. %)
    return new_values

def preprocess_text_simple(df=None, col='', source_values=[], value_default=""): 
    """
    Assuming that the input source values are strings, this function 
    converts all NaNs to empty strings, numeric values to their string counterparts, 
    and remove redundant spaces in the front and back of the source values.  

    """
    return text_processor.preprocess_text_simple(df=df, col=col, source_values=source_values, value_default=value_default)

def remove_duplicates(s, sep=" "):
    tokens = s.split(sep)
    return sep.join(sorted(set(tokens), key=tokens.index))

def join_feature_names(cols, sep='_'): 
    return remove_duplicates(sep.join(cols), sep=sep)

def conjoin0(source_values, sep=" ", remove_dup=False): 
    import pandas as pd

    # new_value = sep.join([str(source_value).strip() for source_value in source_values])
    new_values = []
    for source_value in source_values: 
        if not pd.isna(source_value): 
            new_values.append(str(source_value).strip())
    new_value = sep.join(new_values)

    if remove_dup: 
        tokens = new_value.split(sep)
        new_value = sep.join(sorted(set(tokens), key=tokens.index))
    return new_value

def conjoin(df, cols=[], transformed_vars_only=True, sep=" ", remove_dup=False, col_output=''):
    # combine text data from across multiple columns
    
    if len(cols) > 1: 
        new_values = df[cols].astype(str).agg(sep.join, axis=1).values
    elif len(cols) == 1: 
        new_values = df[cols].values
    else: 
        msg = "(conjoin) Warning: No target columns specified! cols={}\n".format(cols)
        raise ValueError(msg)
    # ... A Series

    if remove_dup: 
        new_values_processed = []
        for new_value in new_values: 
            tokens = new_value.split(sep)
            new_values_processed.append(sep.join(sorted(set(tokens), key=tokens.index)))
        new_values = new_values_processed

    if transformed_vars_only: 
        if len(col_output) > 0: 
            df[col_output] = new_values
            return df
        return new_values

    if len(col_output) == 0: col_output = '_'.join(cols)
    df[col_output] = new_values
    return df    

def predicate_scaling(X, cols=[], mode='minmax'):
    # from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
    # a very rough estimate but if STDs are not approximately equal, we should apply scaling
    v_max = np.max(X, axis=0)
    v_min = np.min(X, axis=0)

    if np.sum(v_max > 1): 
        return True
    if np.sum(v_min < 0): 
        return True 

    return False

def date_of_birth_to_age(x):
    """
    
    Use 
    ---
    df['patient_date_of_birth'].apply(date_of_birth_to_age) 

    Precondition
    ------------
    All values associated with this column has been imputed, 
    reasonable default values filled
    """
    import datetime
    now = datetime.datetime.now()
    return now.year-int(x)

def to_age(df, col='patient_date_of_birth', new_col='age', add_new_col=True, throw_=False, default_val=None):
    if not col in df.columns: 
        msg = "Error: Missing {}".format(col)
        if throw_: raise ValueError(msg)
            
        # noop
        return df 
    import datetime
    now = datetime.datetime.now()
    
    # date_of_path is rarely NaN but it happens
    if default_val is None: default_val = int(df[col].mean())
    df[col].fillna(value=default_val, inplace=True)
    
    df[new_col] = df[col].apply(lambda x: now.year-int(x))
    if add_new_col: 
        pass
    else: 
        df.drop(col, axis=1, inplace=True)
    return df

def resolve_duplicate(df, cols=['test_result_loinc_code', ], add_count=True, col='count'): 
    """
    Count number of duplicate rows, add the count and drop

    Memo
    ----
    1. other possible columns 
       cols = ['test_result_name', 'test_order_name', ]
       
    """
    import pandas as pd

    # drop row-wise duplicates
    n0 = df.shape[0]
    df = df.drop_duplicates(keep='last')
    n1 = df.shape[0]
    print("(resolve_duplicate) Absolute duplicates | n0: {} =?= n1: {}".format(n0, n1))

    if add_count: 
        # df = df.groupby(list(df.columns)).size().reset_index(name=col)
        # ... side effect: the ordering of the rows will change 

        dfx = []
        counts = []

        if len(cols) == 0: 
            # in this case, the notion of "duplicates" is ...
            # ... established by the entire row (i.e. any two rows with exactly the same content are considered as duplicates)
            cols = list(df.columns)
        else: 
            if isinstance(cols, str): 
                cols = [cols, ]

        for r, dfe in df.groupby(cols): 
            count = dfe.shape[0]
            counts.append(count)
            dfe['count'] = count

            dfx.append(dfe)

        n = sum(1 for c in counts if c > 1)
        # n0 = df.shape[0]
        # n1 = df.drop_duplicates(subset=cols, keep='last').shape[0]
        print("(resolve_duplicate) Found {} multiple instances wrt cols: {}".format(n, cols))

        if len(dfx) > 0: 
            df = pd.concat(dfx)# 
            df.sort_index(inplace=True)
        else: 
            df['count'] = 1  # this should not happen

    return df

def dehyphenate(df, col='test_result_loinc_code', drop_cbit=False): # 'LOINC_NUM'
    import loinc 
    return loinc.dehyphenate(df, col=col, drop_cbit=drop_cbit)

def dequote(df, col='Medivo Test Result Type'):
    import loinc
    return loinc.dequote(df, col=col)

def trim_tail(df, col='test_result_loinc_code', delimit=['.', ';']):
    df[col] = df[col].str.lower().replace('(\.|;)[a-zA-Z0-9]*', '', regex=True)
    return df 

def replace_values(df, values=['.', ], new_value='Unknown', col='test_result_loinc_code'):
    for v in values: 
        df[col] = df[col].str.lower().replace(v, new_value)
    
    # ... df[col].str.lower().replace(, 'Unknown') => 'unknown'
    df[col] = df[col].replace(new_value.lower(), new_value) # correction
    return df

# [todo] This is LOINC-specific cleaning operation
def canonicalize(df, col_target="test_result_loinc_code", 
        token_default='unknown', token_missing='unknown', token_other='other', 
        target_labels=[], noisy_values=[], columns=[], verbose=1):
    if not noisy_values: noisy_values = ['request', 'no loinc needed', '.', ';', 'coumt', 'unloinc']
    
    if verbose: 
        print("(canonicalize) Operations> fillna, dehyphenate, replace_values, trim_tail, fill_others")
    
    # fill na 
    df[col_target].fillna(value=token_missing, inplace=True)

    dehyphenate(df, col=col_target)
    replace_values(df, values=noisy_values, new_value=token_missing, col=col_target)
    trim_tail(df, col=col_target, delimit=['.', ';', ])

    df[col_target].replace('', token_missing, inplace=True) 

    # codes that are not in the target set
    if len(target_labels) > 0: 
        if verbose: print("(canonicalize) Focus only on target labels (n={}), labeling the rest as {}".format(len(target_labels), token_other))
        df.loc[~df[col_target].isin(target_labels), col_target] = token_other

    # subset columns (e.g. useful for adding additional data)
    if len(columns) > 0: 
        return df[columns]

    return df
def transform(df, col_target="test_result_loinc_code", token_default='unknown', noisy_values=[]):
    raise NotImplementedError

def get_eff_values(df, col=''):
    if isinstance(y, DataFrame):
        assert col in df.columns
        return list(df[df[col].notnull()][col].values)
    else: 
        # df is a numpy array
        assert isinstance(df, np.ndarry)
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

def take(n, iterable):
    from itertools import islice
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

# --- Complex Transfomration ---
###################################################

def t_transformation(**kargs):
    from analyzer import load_data

    df = load_data(input_file='andromeda-pond-hepatitis-c.csv', warn_bad_lines=False)
    print("> input dim(df): {}".format(df.shape))
    df = resolve_duplicate(df)

    return 

def test(**kargs):

    # test all the feature-transformation functions  
    t_transformation()

    return

if __name__ == "__main__": 
    test()