# -*- coding: utf-8 -*-

import os
from pandas import DataFrame 
import numpy as np
import pandas as pd
import loinc
from loinc import LoincTable, LoincTSet
from loinc import dehyphenate, dequote, is_canonicalized, replace_values
from collections import defaultdict
import data_processor as dp

class LoincMTRT(object):
    # header = ['Test Result LOINC Code', 'Medivo Test Result Type']
    col_key = col_code = "Test Result LOINC Code" # use the loinc code as key even though we are primarily interested in predicting loinc from mtrt
    col_value = "Medivo Test Result Type"

    header = [col_code, col_value]

    table = 'loinc-leela.csv'
    table_prime = 'loinc-leela-derived.csv'

    stop_words = ["IN", "FROM", "ON", "OR", "OF", "BY", "AND", "&", "TO", "BY", "", " "]
    delimit = ','

    # derived features 
    col_joined = "LOINC_MTRT"
    col_unknown = 'unknown'

    # what constitutes the essential information for a given LOINC code i.e. which parts should go into the training corpus 
    # in a NLP model such as a TF-IDF model? 
    col_sn = LoincTable.col_sn
    col_ln = LoincTable.col_ln
    cols_6p = LoincTable.cols_6p
    
    cols_descriptor = [col_sn, col_ln] + cols_6p + [col_value, ]

    @staticmethod
    def load_table(**kargs):
        """
        
        Memo
        ----
        1. In Leela's loinc to mtrt map, both the dehyphenated and hyphenized code
           have their own entry (e.g. both 12345-7, 123457 have an entry with same content)
        2. It's possible that the same code has different MTRT strings

        > Code: 197517 | count=2
  + [1] 197517
      LN:   Mouse epithelium, serum proteins + urine proteins IgE Ab [Units/volume] in Serum
      MTRT: Mouse epithelium
  + [2] 197517
      LN:   Mouse epithelium, serum proteins + urine proteins IgE Ab [Units/volume] in Serum
      MTRT: Mouse epithelium, serum proteins + urine proteins IgE Ab [Units/volume] in Seru

        """
        from transformer import dehyphenate
        sep = LoincMTRT.delimit # kargs.get('sep', ',')
        input_dir = kargs.get('input_dir', 'data')
        dehyphen = kargs.get('dehyphenate', True)
        deq = kargs.get('dequote', True)
        one_to_one = kargs.get('one_to_one', True)

        df = dp.load_generic(input_file=LoincMTRT.table, sep=sep, input_dir=input_dir) 
        if dehyphen: 
            df = dehyphenate(df, col=LoincMTRT.col_key)  # inplace
            # 12345-7 or 123457 
            df = df.drop_duplicates(keep='last')  # drop duplicates

        if deq: 
            df = dequote(df, col=LoincMTRT.col_value)

        if one_to_one: 
            df = LoincMTRT.resolve_duplicates(df, verbose=1)

        return df
    @staticmethod
    def save_loinc_to_mtrt(df, **kargs):

        return

    @staticmethod
    def resolve_duplicates(df, verbose=1, **kargs): 
        """
        
        Memo
        ----
        1. .load_table() loads the mapping (loinc to mtrt), some of which have "duplicate entries"
            i.e. multiple MTRTs for the same code, which one to use? 

            The objective is to come to a 1-to-1 mapping from LOINC code to MTRT

        """
        dfx = []
        codes_multirow = []
        # criterion = kargs.get('criterion', 'length')

        N0 = df.shape[0]
        col_value = LoincMTRT.col_value

        for code, dfe in df.groupby([LoincMTRT.col_code, ]): 
            n = dfe.shape[0]

            if n == 1: 
                dfx.append(dfe)
            else: 
                codes_multirow.append(code)
                
                # --- determine which row to use

                col_new = 'length'
                assert not col_new in dfe.columns
                dfe[col_new] = dfe[col_value].apply(len)
                dfe = dfe.loc[dfe[col_new]==np.max(dfe[col_new])].iloc[[0]]  # use double bracket to keep as dataframe
                
                assert dfe.shape[0] == 1
                dfx.append(dfe)

        df = pd.concat(dfx, ignore_index=True)
        print("(resolve_duplicates) sample size before/after: {} -> {}".format(N0, df.shape[0]))

        return df

    @staticmethod
    def save_derived_loinc_to_mtrt(df, **kargs):
        sep = kargs.get('sep', ',')
        output_dir = kargs.get('output_dir', 'data')
        output_file = kargs.get("output_file", LoincMTRT.table_prime)
        dehyphen = kargs.get('dehyphenate', True)
        deq = kargs.get('dequote', True)

        if dehyphen: 
            df = dehyphenate(df, col=LoincMTRT.col_key)  # inplace
        if deq: 
            df = dequote(df, col=LoincMTRT.col_value)

        dp.save_generic(df, sep=sep, output_file=output_file, output_dir=output_dir) 
        return  
    @staticmethod
    def load_derived_loinc_to_mtrt(**kargs):
        sep = kargs.get('sep', ',')
        input_dir = kargs.get('input_dir', 'data')
        input_file = kargs.get("input_file", LoincMTRT.table_prime)

        # [output[] None if file not found
        return dp.load_generic(input_file=input_file, sep=sep, input_dir=input_dir) 

    @staticmethod
    def transform():
        """
        Transform the LOINC-to-MTRT table (generated from leela) such that the column names 
        are in lowcase with words separated by underscores. 

        e.g. Test Result LOINC Code => test_result_loinc_code

        Additionally, can add additional attributes derived from the baseline columns (e.g. word embedding, 
        term mappings, etc)

        """
        pass
### end LoincMTRT class 

def load_loinc_to_mtrt(input_file='', **kargs):
    if input_file: LoincMTRT.table = input_file
    return LoincMTRT.load_table(**kargs)

def merge_mtrt_loinc_table(df_mtrt=None, df_loinc=None, dehyphenate=True, target_cols=[]):
    # from loinc import LoincMTRT, LoincTable

    # --- LOINC table attributes
    col_ln, col_sn = LoincTable.long_name, LoincTable.short_name
    col_lkey = LoincTable.col_key
    ##################################
    # --- Loinc to MTRT attributes (from Leela)
    col_mval = LoincMTRT.col_value
    col_mkey = LoincMTRT.col_key    # loinc codes in the mtrt table

    if df_mtrt is None: df_mtrt = LoincMTRT.load_table(dehyphenate=dehyphenate, one_to_one=True) 

    # test
    Nu = len(df_mtrt[col_mkey].unique())
    assert Nu == df_mtrt.shape[0], "Some loinc codes do not map to unique MTRT | Nu: {} <> N: {}".format(Nu, df_mtrt.shape[0])

    if df_loinc is None: df_loinc = loinc.load_loinc_table(dehyphenate=dehyphenate)

    # codeSet = set(df_loinc[col_lkey].values).union(df_mtrt[col_mkey])

    df = pd.merge(df_loinc, df_mtrt, left_on=col_lkey, right_on=col_mkey, how='left').drop([col_mkey,], axis=1).fillna('')
    # use 'LOINC_NUM' to refer to the LOINC code

    # test 
    N0 = df.shape[0]
    Nu = len(df[col_lkey].unique())
    assert Nu == N0, "n(unique): {} but size(df)={} | multiple rows for the same code?".format(Nu, N0)
    # if Nu != N0: 
    #     print("(merge_mtrt_loinc_table) n(unique): {} but size(df)={} | multiple rows for the same code?".format(Nu, N0))
    #     n_multirow = 0
    #     # for code, count in dict(df[col_lkey].value_counts()): 
    #     for code, dfc in df.groupby([col_lkey, ] ): 
    #         count = dfc.shape[0]
    #         if dfc.shape[0] > 1:
    #             if n_multirow < 10:  
    #                 print("> Code: {} | count={}".format(code, count)) 
    #                 for i, (r, row) in enumerate(dfc.iterrows()): 
    #                     print("  + [{}] {}".format(i+1, code))
    #                     print("      LN:   {}".format(row[col_ln]))
    #                     print("      MTRT: {}".format(row[col_mval]))
    #             n_multirow += 1
    #     print("... n(unique): {}, n_multirow: {}".format(Nu, n_multirow))
    #     sys.exit(0)

    if len(target_cols) > 0: 

        # we may need this column in order to faciliate join operations later 
        if not col_lkey in target_cols: target_cols = [col_lkey, ] + target_cols

        return df[target_cols]

    return df

def get_corpora_from_merged_loinc_mtrt(df_mtrt=None, df_loinc=None, target_cols=[], dehyphenate=True, 
        remove_dup=False, return_dataframe=False, **kargs): 
    """

    Memo
    ----
    1. Used to generate part of the training corpus
    2. When checking consistency between T-attributes and LOINC descriptors, ... 
    """
    import transformer as tr

    # optional parameters 
    output_file = kargs.get('output_file', LoincTSet.file_merged_loinc_mtrt) # "loinc_mtrt.corpus"
    output_dir = kargs.get("output_dir", "data")
    col_new = kargs.get('col_new', LoincMTRT.col_joined)
    sep = kargs.get('sep', " ")
    save = kargs.get('save', True)
    # load = kargs.get('load', True)  # load precomputed table


    # --- LOINC table attributes
    col_ln, col_sn = LoincTable.long_name, LoincTable.short_name
    col_lkey = LoincTable.col_key
    ##################################
    # --- Loinc to MTRT attributes (from Leela)
    col_mval = LoincMTRT.col_value
    # col_mkey = LoincMTRT.col_key    # loinc codes in the mtrt table

    if not target_cols: 
        target_cols = LoincMTRT.cols_descriptor # SN+LN + 6p + MTRT

    df_merged = merge_mtrt_loinc_table(df_mtrt=df_mtrt, df_loinc=df_loinc, dehyphenate=dehyphenate, target_cols=target_cols)
    corpora = tr.conjoin(df_merged, cols=target_cols, transformed_vars_only=True, sep=sep, remove_dup=remove_dup)
    if save: 
        # df = DataFrame(corpora, columns=[col_new, ])
        df_merged[col_new] = corpora
        output_path = os.path.join(output_dir, output_file)
        print("(corpora_from_merged_loinc_mtrt) Saving merged LOINC and MTRT dataframe (cols={}) to:\n{}\n".format(df_merged.columns, 
            output_path))
        df_merged.to_csv(output_path, sep=",", index=False, header=True)

    if return_dataframe: 
        df_merged[col_new] = corpora
        # df_merged[col_lkey] = df_merged[col_lkey].astype(str)
        return df_merged

    return corpora

def load_corpora_from_merged_loinc_mtrt(**kargs):
    input_file = kargs.get('input_file', LoincTSet.file_merged_loinc_mtrt) # "loinc_mtrt.corpus"
    input_dir = kargs.get("input_dir", "data")
    sep = kargs.get('sep', ",")
    verbose = kargs.get('verbose', 1)

    df = None
    input_path = os.path.join(input_dir, input_file)
    if os.path.exists(input_path) and os.path.getsize(input_path) > 0:
        df = pd.read_csv(input_path, sep=sep, header=0, index_col=None, error_bad_lines=False)
        if verbose: print("(corpora_from_merged_loinc_mtrt) Loaded loinc-mtrt corpus (dim={}) from:\n{}\n".format(df.shape, input_path))
    else: 
        if verbose: print("(corpora_from_merged_loinc_mtrt) corpus file does not exist yet at:\n{}\n".format(input_path))

    return df

def get_loinc_corpus_lookup_table(dehyphenate=True, remove_dup=False, verify=True):  

    # --- LOINC table attributes
    col_ln, col_sn = LoincTable.long_name, LoincTable.short_name
    col_lkey = LoincTable.col_key
    # --- MTRT table
    col_mval = LoincMTRT.col_value

    dflm = load_corpora_from_merged_loinc_mtrt()
    if dflm is None: 
        dflm = get_corpora_from_merged_loinc_mtrt(target_cols=[col_sn, col_ln, col_mval], 
                        dehyphenate=dehyphenate, remove_dup=remove_dup, return_dataframe=True)
    dflm[col_lkey] = dflm[col_lkey].astype(str)
    # ... need to ensure that the LOINC codes are of string type

    loinc_lookup = dflm.set_index(col_lkey)[LoincMTRT.col_joined].to_dict()   # LoincTSet.col_corpus
    if verify: 
        from utils_sys import sample_dict
        tb = sample_dict(loinc_lookup, n_sample=10)
        assert np.all([isinstance(k, str) and isinstance(v, str) for k, v in tb.items()]), \
             "Invald dtype found in loinc_lookup:\n{}\n".format(tb)

    return loinc_lookup

def get_loinc_descriptors(dehyphenate=True, remove_dup=False, verify=True, verbose=1, recompute=False):
    # from utils_sys import sample_dict

    # --- LOINC table attributes
    col_com = LoincTable.col_com 
    col_ln, col_sn = LoincTable.long_name, LoincTable.short_name
    col_lkey = LoincTable.col_key
    cols_6p = LoincTable.cols_6p
    # --- MTRT table
    col_joined = LoincMTRT.col_joined
    col_mval = LoincMTRT.col_value
    col_mkey = LoincMTRT.col_key    # loinc codes in the mtrt table

    target_cols = LoincMTRT.cols_descriptor # + [col_joined, ] # SN + LN + 6p + MTRT + corpus
    
    dflm = load_corpora_from_merged_loinc_mtrt()
    tFullInfo = np.all(col in dflm.columns for col in target_cols)
    if recompute or (dflm is None or not tFullInfo): 
        if verbose: print("(get_loinc_descriptors) Recomputing LOINC descriptors ...")
        dflm = get_corpora_from_merged_loinc_mtrt(target_cols=target_cols, 
                        dehyphenate=dehyphenate, remove_dup=remove_dup, return_dataframe=True)
    dflm[col_lkey] = dflm[col_lkey].astype(str)

    # ... need to ensure that the LOINC codes are of string type
    assert np.all([col in dflm.columns for col in target_cols]), "col(dflm):\n{}\ncol(expected):\n{}\n".format(dflm.columns, target_cols)
    
    # loinc_lookup = defaultdict(dict)
    loinc_lookup = {}
    target_cols += [col_joined, ] # include the corpus column as well
    for r, row in dflm.iterrows(): 
        code = row[col_lkey]
        if not code in loinc_lookup: loinc_lookup[code] = {col: "" for col in target_cols}
        for col in target_cols: 
            loinc_lookup[code][col] = row[col]

    if verify: 
        from utils_sys import sample_dict
        tb = sample_dict(loinc_lookup, n_sample=10)
        # codes_missed = ['882934', ] # codes not found in LOINC table
        # for k, v in tb.items(): 
        #     assert isinstance(k, str)
        #     print("... [{}]\n{}\n".format(k, v))
        #     assert np.all([col in v for col in target_cols])
        assert np.all([isinstance(k, str) and isinstance(v[col_joined], str) for k, v in tb.items()]), \
             "Invald dtype found in loinc_lookup:\n{}\n".format(tb)

    return loinc_lookup

def test(): 

    return

if __name__ == "__main__": 
    test()