# -*- coding: utf-8 -*-

import os, sys
import re
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

    # print("(merge_mtrt_loinc_table) target_cols:{}\n... cols(merged):\n{}\n".format(target_cols, df.columns))
    # ['SHORTNAME', 'LONG_COMMON_NAME', 'COMPONENT', 'PROPERTY', 'TIME_ASPCT', 'SYSTEM', 'SCALE_TYP', 'METHOD_TYP', 'Medivo Test Result Type']
    if len(target_cols) > 0: 

        # we may need this column in order to faciliate join operations later 
        if not col_lkey in target_cols: target_cols = [col_lkey, ] + target_cols
        if LoincMTRT.col_joined in target_cols: 
            print("(merge_mtrt_loinc_table) Warning: Somehow LOINC_MTRT attribute seeped into target_cols:\n{}\n".format(target_cols))
            target_cols.remove(LoincMTRT.col_joined)

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
    col_new = kargs.get('col_new', LoincMTRT.col_joined) # LOINC_MTRT
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

    if not target_cols: target_cols = LoincMTRT.cols_descriptor # SN+LN + 6p + MTRT ([col_sn, col_ln] + cols_6p + [col_value, ])
    print("(corpora_from_merged_loinc_mtrt) target cols:\n{}\n".format(target_cols))
    if LoincMTRT.col_joined in target_cols: target_cols.remove(LoincMTRT.col_joined)

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
        return df_merged   # col_new (e.g. LOINC_MTRT) carries the combined document of LOINC descriptors and MTRT

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
    """
    Get the mapping (as a dictionary) from LOINC code to its descriptors (i.e. selective columns from the standard LOINC table)

    """
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

#################################################################
# --- Utilities for feature extractions 

def extract_slots(df=None, col='', source_values=[], **kargs):
    verbose = kargs.get('verbose', 0)
    docType = kargs.get('doc_type', "long name")
    remove_slot = kargs.get('remove_slot', False) # if True, remove slot from text
    save = kargs.get('save', False)

    if len(source_values) > 0: 
        if not col: col = 'processed'
        df = DataFrame(source_values, columns=[col, ])
    else: 
        assert df is not None, "Both input dataframe (df) and source values were not given!"
        source_values = df[col].values

    # precondition
    Nvar = df.shape[1]

    # brackets
    if verbose: print("(extract_slots) Extracting measurement units (i.e. [...]) ... ")
    cols_target = [col, "unit"]
    col_unit = 'unit'
    cols_derived = ['unit', ]
    bracketed = []
    token_default = ""
    null_rows = []
    # n_malformed = 0
    malformed = []
    for r, doc in enumerate(source_values): 
        if pd.isna(doc) or len(str(doc)) == 0: 
            null_rows.append(r)
            bracketed.append(token_default)
            continue

        b, e = doc.find("["), doc.find("]")
        if b > 0: 
            if not e > b: 
                if verbose: print("(extract_slots) Weird doc (multiple [])? {}".format(doc))
                # n_malformed += 1
                malformed.append(doc)
                bracketed.append(token_default)
                continue

            bracketed.append( re.search(r'\[(.*?)\]',doc).group(1).strip() )   # use .*? for non-greedy match
        else: 
            bracketed.append(token_default)
    null_rows = set(null_rows)

    
    # --------------------------------------
    # derived attributes
    df[col_unit] = bracketed
    # df[col_unit] = df[col].apply(re.search(r'\[(.*?)\]',s).group(1))
    # --------------------------------------

    if remove_slot: 

        # [test]
        target_index = []
        if verbose: 
            dft = df[df[col].str.contains("\[.*?\]")]
            target_index = dft.index

            print("(extract_slots) Malformed []-terms (n={}):\n{}\n".format(len(malformed), display(malformed)))
            print("(extract_slots) After extracting unit (doc type: {}) | n(has []):{}, n(malformed): {} ...\n{}\n".format(docType, 
                    dft.shape[0], len(malformed),
                        tabulate(dft[cols_target].head(20), headers='keys', tablefmt='psql')))

        df[col] = df[col].str.replace("\[.*?\]", '')

        if verbose and len(target_index) > 0: 
            print("(extract_slots) After removing brackets:\n{}\n".format(tabulate(df.iloc[target_index][cols_target].head(20), headers='keys', tablefmt='psql')))

    ########################################################

    # parenthesis
    abbreviations = []
    compounds = []
    new_docs = []
    col_abbrev = 'abbrev'
    col_comp = 'compound'
    cols_derived = cols_derived + [col_abbrev, col_comp, ]

    ########################################################
    p_abbrev = re.compile(r"(?P<compound>[-+a-zA-Z0-9,']+)\s+\((?P<abbrev>.*?)\)")  # p_context
    # ... capture 2,2',3,4,4',5-Hexachlorobiphenyl (PCB)
    # ... cannot capture Bromocresol green (BCG)

    p_aka = re.compile(r"(?P<compound>[-+a-zA-Z0-9,']+)/(?P<abbrev>[-a-zA-Z0-9,']+)")
    # ... 3-Hydroxyisobutyrate/Creatinine

    p_abbrev2 = re.compile(r"(?P<compound>([-+a-zA-Z0-9,']+)(\s+[-a-zA-Z0-9,']+)*)\s+\((?P<abbrev>.*?)\)")
    # ... von Willebrand factor (vWf) Ag actual/normal in Platelet poor plasma by Immunoassay

    p_by = re.compile(r".*by\s+(?P<compound>([-+a-zA-Z0-9,']+)(\s+[-a-zA-Z0-9,']+)*)\s+\((?P<abbrev>.*?)\)")
    p_supplement = p_ps = re.compile(r".*--\s*(?P<ps>([-+a-zA-Z0-9,']+)(\s+[-a-zA-Z0-9,']+)*)")
    p_context_by = re.compile(r"by\s+(?P<compound>([-+a-zA-Z0-9,']+)(\s+[-a-zA-Z0-9,']+)*)\s+\((?P<abbrev>.*?)\)")
    ########################################################
    
    if verbose: print("(extract_slots) Extracting compounds and their abbreviations ... ")
    cols_target = [col, "compound", "abbrev"]
    token_default = ""
    n_null = n_malformed = 0
    malformed = []
    for r, doc in enumerate(source_values): 
        # if r in null_rows: 
        #     abbreviations.append(token_default)
        #     compounds.append(token_default)
        #     continue

        b, e = doc.find("("), doc.find(")")
        tHasMatch = False
        if b > 0: 
            if not (e > b): 
                if verbose: print("(extract_slots) Weird doc (multiple parens)? {}".format(doc))
                # e.g. MTRT: Functional Assessment of Incontinence Therapy - Fecal Questionnaire - version 4 ( [FACIT]
                #      missing closing paran
                abbreviations.append(token_default)
                compounds.append(token_default)
                # n_malformed += 1
                malformed.append(doc)
                new_docs.append(doc)
                continue

            m = p_abbrev.match(doc)
            if m: 
                abbreviations.append(m.group('abbrev').strip())
                compounds.append(m.group('compound').strip())
                tHasMatch = True
            else:
                # long name followed by keyword "by"
                m = p_by.match(doc)
                # e.g. fusion transcript  in Blood or Tissue by Fluorescent in situ hybridization (FISH) Narrative
                # ~> Fluorescent in situ hybridization (FISH)
                if m: 
                    abbreviations.append(split_and_strip(m.group('abbrev'))) # m.group('abbrev').strip()
                    compounds.append(split_and_strip(m.group('compound'))) # m.group('compound').strip())
                    tHasMatch = True
                else: 
                    m = p_abbrev2.match(doc)
                    if m: 
                        abbreviations.append(split_and_strip(m.group('abbrev')))
                        compounds.append(split_and_strip(m.group('compound')))
                        tHasMatch = True

            if not tHasMatch: 
                # not matched in the beginning
                # e.g. 14-3-3 protein [Presence] in Cerebral spinal fluid by Immunoblot (IB)
                m = p_context_by.search(doc)
                if m: 
                    abbreviations.append(split_and_strip(m.group('abbrev')))
                    compounds.append(split_and_strip(m.group('compound')))
                    tHasMatch = True

        ########################
        if not tHasMatch: 
            d = doc.find("/")
            if d > 0: 
                m = p_aka.match(doc)
                if m: 
                    abbreviations.append(m.group('abbrev').strip())
                    compounds.append(m.group('compound').strip())
                    tHasMatch = True

        ########################
        if not tHasMatch: 
            abbreviations.append(token_default)
            compounds.append(token_default)

            # doc: no change
        else:
            # [test]
            if remove_slot: 
                doc = re.sub('\(.*?\)', '', doc)
 
                b, e = doc.find("("), doc.find(")")
                assert b < 0 or e < 0, "Multiple parens? {}".format(doc)

        new_docs.append(doc)
            
    # --------------------------------------
    # derived attributes
    df[col_comp] = compounds
    df[col_abbrev] = abbreviations
    # --------------------------------------

    if remove_slot:

        # [test]
        target_index = []
        if verbose: 
            dft = df[df[col].str.contains("\(.*?\)")]
            target_index = dft.index
            print("(extract_slots) Malformed ()-terms (n={}):\n{}\n".format(len(malformed), display(malformed)))
            print("(extract_slots) After extracting 'compound' & 'abbreviation' (doc type: {}) | n(has_paran):{}, n(malformed):{} ...\n{}\n".format(
                docType, dft.shape[0], len(malformed),
                    tabulate(dft[cols_target][[col, col_abbrev]].head(200), headers='keys', tablefmt='psql')))

        df[col] = df[col].str.replace("\(.*?\)", '')

        if verbose > 1 and len(target_index) > 0: 
            print("(extract_slots) After removing parens:\n{}\n".format(tabulate(df.iloc[target_index][cols_target].head(100), headers='keys', tablefmt='psql')))
    # -------------------------------------------------------------        
    # complex cases: 
    #    Hepatitis B virus DNA [log units/volume] (viral load) in Serum or Plasma by NAA with probe detection

    # df[col] = new_docs
    ########################################################

    if verbose: print("(extract_slots) Extracting Postscript ... ")
    cols_target = [col, "note"]
    col_note = 'note'
    cols_derived = cols_derived + [col_note, ]
    token_default = ""
    notes = []
    for r, doc in enumerate(source_values): 
        # if r in null_rows: 
        #     notes.append(token_default)
        #     continue

        m = p_ps.match(doc)
        if m: 
            notes.append(split_and_strip(m.group('ps')))
        else: 
            notes.append(token_default)

    # --------------------------------------
    df[col_note] = notes
    # --------------------------------------
    if remove_slot: 
        df[col] = df[col].str.replace("--", " ")

    if verbose > 1: 
        dft = df[df[col].str.contains("--.*")]
        print("(extract_slots) After extracting additional info (PS) [doc type: {}] | n(has PS): {} ... \n{}\n".format(
            docType, dft.shape[0], tabulate(dft[cols_target].head(50), headers='keys', tablefmt='psql')))

    assert df.shape[1] == Nv + 4

    if save: 
        # output_file=kargs.get('output_file')
        LoincMTRT.save_derived_loinc_to_mtrt(df)

    return df


def test(): 

    return

if __name__ == "__main__": 
    test()