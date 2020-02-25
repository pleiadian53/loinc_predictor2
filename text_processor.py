# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np
from pandas import DataFrame, Series
import os, re, sys, time
import random
from tabulate import tabulate
from collections import defaultdict, Counter


def has_common_tokens(s1, s2):
    if pd.isna(s1) or pd.isna(s2): 
        return False 
    sv1 = str(s1).split()
    sv2 = str(s2).split()
    if len(set(sv1).intersection(sv2)) > 0: 
        return True
    return False

def has_common_prefix(s1, s2, n=1): 
    if pd.isna(s1) or pd.isna(s2): 
        return False 
    if len(s1) > 0 and len(s2) > 0: 
        sv1 = np.array(str(s1).split())
        sv2 = np.array(str(s2).split())
        # print("... sv1: {}\n... sv2: {}".format(sv1, sv2))
        return np.all(sv1[:n] == sv2[:n])
    return False

def preprocess_text_simple(df=None, col='', source_values=[], value_default=""): 
    """
    Assuming that the input source values are strings, this function 
    converts all NaNs to empty strings, numeric values to their string counterparts, 
    and remove redundant spaces in the front and back of the source values.  

    """
    # import pandas as pd
    
    hasValidDf = df is not None and col in df.columns
    if isinstance(source_values, str): source_values = [source_values, ]
    if len(source_values) == 0: # unique_tests
        assert hasValidDf, "Neither the source values nor training data were given!"
        source_values = df[col].values 
        
    source_values_processed = []
    n_null = n_numeric = 0
    for source_value in source_values: 
        if pd.isna(source_value): 
            source_values_processed.append(value_default)
            n_null += 1
        elif isinstance(source_value, (int, float, )): 
            n_numeric += 1
            source_values_processed.append(str(source_value))
        else: 
            source_values_processed.append( source_value.strip() )

    if hasValidDf: 
        df[col] = source_values_processed
        return df 
    return source_values_processed
# --- alias ---
preprocess_text = preprocess_text_simple

def process_text(df=None, col='', source_values=[], clean=True, standardized=True, **kargs): 
    """
    Parse long strings, such as LOINC's long name field and MTRT, which typically serve as "tags"

    Operations: 1. preprocess the input/source values so that each value is guranteed to be in string format (e.g. NaN turned into
                   empty string, floats and ints into their string forms)
                2. Simple slot filling identified by brackets, parens, dashes, etc.
                3. text cleaning by removing punctuations

    Note that the name of the slots/derived attributes may not be always what they meant to be used. 
    e.g. 'unit' typically refers to measurement units but sometimes other kinds of values could be enclosed within brackets as well. 

    """
    def split_and_strip(s): 
        return ' '.join([str(e).strip() for e in s.split()])
    
    from CleanTextData import clean_term, standardize

    # for debugging and testing only
    verbose = kargs.get('verbose', 0)
    # docType = kargs.get('doc_type', "long name")
    value_default = kargs.get("value_default", "")   # default value for NaN/Null
    return_dataframe = False
    # save = kargs.get('save', False)
    # transformed_vars_only= kargs.get('transformed_vars_only', True)
    
    if not isinstance(source_values, (list, np.ndarray)): source_values = [source_values, ]
    if len(source_values) > 0: 
        return_dataframe = False
    else: 
        assert df is not None, "Both the dataframe (df) and source values were not given!"
        source_values = df[col].values
        return_dataframe = True

    # preprocess source value to ensure that all values of in string type
    source_values = preprocess_text_simple(source_values=source_values, value_default=value_default)

    ########################################################

    if standardized: # this has to come before clean operation
        # df[col] = df[col].apply(standardize)
        source_values = [standardize(source_value) for source_value in source_values]

    # clean text 
    if clean: 
        # dft = df.loc[df[col].str.contains(r'\bby\b', flags=re.IGNORECASE)]
        # index_keyword = dft.index
        # print("... Prior to cleaning | df(by):\n{}\n".format(dft.head(10)))

        # df[col] = df[col].apply(clean_term)
        source_values = [clean_term(source_value) for source_value in source_values]

    else: 
        # remove extra spaces
        # df[col] = df[col].apply(split_and_strip)
        source_values = [split_and_strip(source_value) for source_value in source_values]

    if return_dataframe: 
        df[col] = source_values
        return df

    # if save: 
    #     output_file=kargs.get('output_file')
    #     LoincMTRT.save_derived_loinc_to_mtrt(df)
    return source_values

# --- Alias --- 
process_text_col = process_text

def posthoc_process_text(df=None, col='', source_values=[], **kargs):

    # for debugging and testing only
    verbose = kargs.get('verbose', 0)
    # docType = kargs.get('doc_type', "long name")
    value_default = kargs.get("value_default", "")   # default value for NaN/Null
    return_dataframe = False

    if not isinstance(source_values, (list, np.ndarray)): source_values = [source_values, ]
    if len(source_values) > 0: 
        return_dataframe = False
    else: 
        assert df is not None, "Both the dataframe (df) and source values were not given!"
        source_values = df[col].values
        return_dataframe = True

    ###############################################
    # --- add rules here --- 

    ###############################################

    if return_dataframe: 
        df[col] = source_values
        return df

    return source_values





