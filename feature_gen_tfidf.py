import pandas as pd 
import numpy as np
from pandas import DataFrame, Series
import os, re, sys, time
import random
from tabulate import tabulate
from collections import defaultdict, Counter

from scipy.spatial import distance # cosine similarity
from sklearn.base import BaseEstimator, ClassifierMixin

# local modules 
import loinc
from loinc import LoincTable, LoincTSet, FeatureSet, MatchmakerFeatureSet
from loinc_mtrt import LoincMTRT
import loinc_mtrt as lmt

from utils_sys import highlight
from language_model import build_tfidf_model
import config

import common
import text_processor as tproc
from text_processor import process_text, process_string
from CleanTextData import standardize 

# from utils_plot import saveFig # contains "matplotlib.use('Agg')" which needs to be called before pyplot 
# from matplotlib import pyplot as plt

"""
MTRT text processor

Reference
---------
1. Leela: see classification rules that lead to medivo_test_result_type
          
          https://github.com/medivo/leela

Memo
----
* Cleaning texts 

  https://machinelearningmastery.com/clean-text-machine-learning-python/

* String matching and similarity: 

    a. Levenshtein distance 

       pip install python-Levenshtein

    b. td-idf vectorizer

       https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76


Update
------


"""
p_oc = p_organic_compound = re.compile(r'(?P<prefix>[0-9\',]+)-(?P<suffix>\w+)')

class ReadTxtFiles(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), encoding='latin'):
                yield simple_preprocess(line)

class MTRTClassifier(BaseEstimator, ClassifierMixin):  
    """
    Predict LOINC codes by MTRT texts.

    Ref
    ---
    1. customized sklearn estimator: 

            http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/

    """
    def __init__(self, source_table=''):
        """
        Called when initializing the classifier
        """
        self.source_table = source_table
        if not source_table: self.source_table = LoincMTRT.table

        self.table = LoincMTRT.load_table(dehyphenate=True, dequote=True) # input_file/LoincMTRT.table

    def fit(self, X, y=None):
        """

        Memo
        ----
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """

        self.threshold_ = 0.5

        return self

    def to_label(self, x):
        # returns True/False according to fitted classifier
        # notice underscore on the beginning
        return( 1 if x >= self.threshold_ else 0 )

    def predict(self, X, y=None):
        try:
            getattr(self, "threshold_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return([self.to_label(x) for x in X])

    def score(self, X, y=None):
        # counts number of values bigger than mean
        return(sum(self.predict(X))) 

### End class MTRTClassifier

def process_each(mtrt_str, code=''):
    def split_and_strip(s): 
        return ' '.join([str(e).strip() for e in s.split()])
    
    header = LoincMTRT.header  # LoincMTRT
    adict = {h:[] for h in header}
    adict[header[0]].append(str(code))
    adict[header[1]].append(mtrt_str)
    df = DataFrame(adict, columns=adict.keys())

    df = extract_slots(df)
    
    return dict(df.iloc[0])

def process_loinc_table(): 
    pass

# def process_string(s, doc_type='string'): 
#     if pd.isna(s): return ""
#     sp = process_text(source_values=s, clean=True, standardized=True, doc_type=doc_type)[0]
#     return sp

def display(x, n_delimit=80): 
    msg = "#" * n_delimit + '\n'
    if isinstance(x, list): 
        for i, e in enumerate(x): 
            msg += "... [{}] {}\n".format(i, e)
    elif isinstance(x, dict): 
        for k, v in enumerate(x): 
            msg += "... [{}] {}\n".format(k, v)
    msg += "\n" + "#" * n_delimit
    return msg

def extract_slots(df=None, col='', source_values=[], **kargs):
    # [output] dataframe with augmented columns capturing the "slots" (e.g. measurement units)
    return LoincMTRT.extract_slots(df=df, col=col, source_values=source_values, **kargs)

def clean_mtrt(df=None, col_target='medivo_test_result_type', **kargs): 
    """

    Related
    -------
    MapLOINCFields.parse_loinc()
    """
    from CleanTestsAndSpecimens import clean_terms, clean_term

    siteWordCount = defaultdict(Counter)
    mtrtList = defaultdict(list)
    medivo_test_result_type = config.tagged_col

    #######################################
    # --- generic parameters
    cohort = kargs.get('cohort', 'hepatitis-c')    
    col_key = kargs.get('col_key', LoincMTRT.col_key) # 'Test Result LOINC Code'
    save = kargs.get('save', True)
    verbose = kargs.get('verbose', 1)
    sep = kargs.get('sep', ',')
    #######################################
    # --- operational parameters
    add_derived_attributes = kargs.get('add_derived', False)

    if df is None: 
        col_target = ''
        # df = LoincMTRT.load_table(dehyphenate=True, dequote=True) # input_file/LoincMTRT.table

        # default: load source data (training data)
        df = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=True, processed=True)

    # df = process_text(df, col='medivo_test_result_type', add_derived=False, save=False)

    site = config.site
    new_values = []
    for r, row in df.iterrows():
        value = str(row[medivo_test_result_type]).upper().strip()

        if ((row[site] not in mtrtList) or 
                (row[site] in mtrtList and value not in mtrtList[row[site]])):
            mtrtList[row[site]].append(value)

        new_values.append( clean_term(value, site=row[site], siteWordCount=siteWordCount, dataType='medivo_test_result_type') )
    df[col_target] = new_values

    if save: 
        output_path = config.processed_file
        if verbose: print("(clean_mtrt) Saving new training data (with cleaned values) at:\n{}\n".format(output_path))        
        # df = update_values(df)
        df.to_csv(output_path, sep=sep, index=False, header=True)
    
    return df

def get_corpora_by_loinc(df, target_cols, **kargs):
    """
    Obtain a training corpus such that each LOINC code is represented by a 
    single document. 

    Typically multiple values are observed for each LOINC code in the text-valued 
    attribtues  such as test_order_name, test_result_name. That is, a LOINC code 
    can be associated with multiple expressions of lab orders and lab results. To 
    form a single document for each code, we first perform a row-wise merge
    followed by column-wise merge. That is, we first consolidate the values
    for each related column (in target_cols) into a signle value (i.e. 
    by concatenating all text values associated with the same LOINC code), 
    followed by concatenating the text across all target attribute. 

    Memo
    ----
    1. Unlike get_corpora_from_dataframe(), this function group text data 
       by LOINC codes. Some LOINC code may have two or more corresponding 
       T-attributes (e.g. two different test_order_name(s) with the same 
       LOINC code). 

    Related
    -------
    1. analyzer.analyze_hard_cases()
    """
    def summarize(n=20, n_sub=10): 
        for i, (code, entry) in enumerate(cache.items()): 
            if i >= n: break 
            msg = '(summary) Code {} has multivalued attributes:\n'.format(code)
            for j, (col, values) in enumerate(entry.items()): 
                msg += "... {} (nv={}):\n".format(col, len(values))
                for k, value in enumerate(values): 
                    if k >= n_sub: break
                    msg += "    + {}\n".format(value)
            print(msg)
        return

    from loinc import LoincTSet
    import transformer as tr
    # from transformer import conjoin, conjoin0
    col_code = kargs.get('col_code', LoincTSet.col_code)  # test_result_loinc_code
    col_new = kargs.get('col_new', 'corpus')
    remove_dup = kargs.get('remove_dup', False)
    tProcessText = kargs.get('process_text', True) # processed
    verbose = kargs.get('verbose', 1)
    return_dataframe = kargs.get('return_dataframe', False)

    cache = {}
    dfx = []
    # if not col_code in target_cols: target_cols = [col_code, ] + target_cols
    n_codes = 0
    for code, dfc in df.groupby([col_code, ]): 
        if verbose and (n_codes > 0 and n_codes % 10) == 0: 
            print("(corpora_by_loinc) Processing code #{}: {}, n(rows): {}    #####".format(n_codes, code, dfc.shape[0]))

        adict = {col:[] for col in target_cols}
        for col in target_cols: 
            source_values = df[col].str.upper().unique() # dfc[col].unique()
            if tProcessText:
                source_values = process_text(source_values=source_values, clean=True, standardized=True)
                source_values = np.unique(source_values)
                # ... keep only unique values
                source_values = tr.remove_null_like_values(source_values, extended='%')

                if len(source_values) > 1: 
                    if not code in cache: cache[code] = {}
                    cache[code][col] = source_values
            
            adict[col].append(tr.conjoin0(source_values, remove_dup=remove_dup))

        dfc_new = DataFrame(adict, columns=target_cols)
        assert dfc_new.shape[0] == 1
        
        dfc_new[col_code] = code
        dfx.append(dfc_new)
        n_codes +=1

    df = pd.concat(dfx, ignore_index=True)
    if verbose: summarize(n=20)
    print("... After consolidation > dim(df):{}, col(df):\n{}\n".format(df.shape, df.columns))

    Nu = len(df[col_code].unique())
    N = df.shape[0]
    assert N == Nu, "Each loinc code should only have one row with contents of multivalued attributes consolidated N: {} <> Nu: {}".format(N, Nu)
    
    if verbose: print("(corpora_by_loinc) Now generating training corpus after consolidating multivalued columns ...")
    
    N0 = df.shape[0]
    corpora = get_corpora_from_dataframe(df, target_cols, **kargs)
    assert len(corpora) == N0

    if return_dataframe:
        assert not (col_new in df.columns), "Corpus column is not found in the returned dataframe | cols(df):\n{}\n".format(df.columns)
        df[col_new] = corpora 
        return df

    return corpora

def get_corpora_from_dataframe(df, target_cols, **kargs): 
    import transformer as tr

    # add_derived = kargs.get('add_derived', False)
    add_loinc_mtrt = kargs.get('add_loinc_mtrt', True)
    dehyphenated = kargs.get('dehyphenate', True)
    tProcessText = kargs.get('process_text', True)
    verbose = kargs.get('verbose', 1)
    delimit = kargs.get('delimit', " ") 
    value_default = kargs.get('value_default', "") 
    save = kargs.get("save", True) # save corpora
    remove_dup = kargs.get("remove_dup", False)
    output_file = kargs.get("output_file", "test.corpus") 
    output_dir = kargs.get("output_dir", "data")

    ##################################
    # --- training data attributes
    col_code = LoincTSet.col_target # 'test_result_loinc_code'
    col_mtrt = LoincTSet.col_tag    # 'medivo_test_result_type'
    ##################################
    # --- LOINC table attributes
    col_ln, col_sn = LoincTable.long_name, LoincTable.short_name
    col_lkey = LoincTable.col_key
    ##################################
    # --- Loinc to MTRT attributes (from Leela)
    col_mval = LoincMTRT.col_value
    col_mkey = LoincMTRT.col_key

    # if add_loinc_mtrt: 
    #     if not col_code in target_cols: target_cols.append(col_code)
    #     if not col_mtrt in target_cols: target_cols.append(col_mtrt)
        # df_mtrt = kargs.get('df_mtrt', None)
        # df_loinc = kargs.get('df_loinc', None)
        # if df_mtrt is None: df_mtrt = LoincMTRT.load_table(dehyphenate=dehyphenate) 
        # if df_loinc is None: df_loinc = loinc.load_loinc_table(dehyphenate=dehyphenate)
        
    df = df.fillna(value_default)
    corpora = tr.conjoin(df, cols=target_cols, remove_dup=remove_dup, transformed_vars_only=True, sep=delimit)

    if tProcessText: 
        N0 = len(corpora)
        # col_new = 'temp_'
        corpora = process_text(source_values=corpora, clean=True, standardized=True, save=False, verbose=verbose) # col=col_new
        assert len(corpora) == N0

    if add_loinc_mtrt: 
        # pack the existing corpora in a new column 
        col_pt0 = 'temp_'
        assert not col_pt0 in df.columns
        df[col_pt0] = corpora 
        N0 = df.shape[0]
        # ------------------------------------
        col_pt = LoincMTRT.col_joined # 'LOINC_MTRT' 
        cols_descriptor = LoincMTRT.cols_descriptor
        # dfp = lmt.load_corpora_from_merged_loinc_mtrt()

        dfp = lmt.get_corpora_from_merged_loinc_mtrt(dehyphenate=True, sep=delimit, remove_dup=remove_dup, 
                     return_dataframe=True, col_new=col_pt, target_cols=cols_descriptor)
        # ... target_cols <- LoincMTRT.cols_descriptor
        # ... dfp: the combined loinc+mtrt corpus in a new column 'col_pt'
        # assert col_lkey in dfp.columns

        print("(get_corpora_from_dataframe) Example merge LOINC and MTRT:")
        for r, row in dfp.sample(n=10).iterrows(): 
            print("... %s" % row[col_pt])

        dfp = process_text(df=dfp, col=col_pt, clean=True, standardized=True, transformed_vars_only=False)  # transformed_vars_only/True
        # assert col_lkey in dfp.columns
        print("... dim(dfp): {} =>\n{}\n".format(dfp.shape, dfp[[col_lkey, col_sn, col_ln]].head(10).to_string(index=False)))
        Nt = dfp.shape[0]
        Nu = len(dfp[col_lkey].unique())
        assert Nt == Nu, f"Nt: {Nt}, Nu: {Nu}"

        # merge 
        assert col_code in df.columns and col_lkey in dfp.columns
        df = pd.merge(df, dfp, left_on=col_code, right_on=col_lkey, how='left')   
        df = df.fillna(value_default)

        #--------------------------------------------------
        # This may hold true when some loinc codes in dfp have more than one entry
        assert df.shape[0] == N0, "N0: {}, N_merged: {}".format(N0, df.shape[0])
        #--------------------------------------------------
        
        corpora = tr.conjoin(df, cols=[col_pt0, col_pt], remove_dup=remove_dup, transformed_vars_only=True, sep=" ")
        # col_pt0: T-attributes portoin of the document
        # col_pt: merged LOINC-MTRT portion of the document

    if save: 
        output_path = os.path.join(output_dir, output_file)
        df = DataFrame(corpora, columns=['corpus'])
        print("(get_corpora_from_dataframe) Saving copora to:\n{}\n".format(output_path))
        df.to_csv(output_path, index=False, header=False)

    return corpora

def tfidf_pipeline(corpora=[], df_src=None, target_cols=[], **kargs):
    """
    Build a TF-IDF model based on the input corpus
    
    Flow
    ----
    1. define corpus given the source data
    2. train model
    
    """
    import language_model as lm
    import transformer as tr
  
    dehyphenate = kargs.get('dehyphenate', True)
    remove_dup = kargs.get('remove_dup', False)
    process_docs = kargs.get('process_docs', False) # remove punctuatinos, cleaning (+ slot extraction), etc.
    # ... only relevant when given source_values
    ##################################
    # --- training data attributes
    col_code = 'test_result_loinc_code'
    col_mtrt = 'medivo_test_result_type'
    ##################################
    # --- LOINC table attributes
    col_ln, col_sn = LoincTable.long_name, LoincTable.short_name
    col_lkey = LoincTable.col_key
    ##################################
    # --- Loinc to MTRT attributes (from Leela)
    col_mval = LoincMTRT.col_value
    col_mkey = LoincMTRT.col_key

    # --- define training corpus
    # A. user-provided corpus extracted from columns of the dataframe 
    col_new = 'conjoined'

    source_values = corpora  # here "source values" is the training corpus
    if len(source_values) > 0: 
        n = len(source_values)
        assert np.all([isinstance(val, str) for val in np.random.choice(source_values, min(n, 10))])

        if process_docs: 
            source_values = \
                 process_text(source_values=source_values, 
                    clean=True, standardized=True, save=False, doc_type='training data')
        # source_values = dfp[col_new].values
    elif df_src is not None: 
        assert len(target_cols) > 0, "Target columns must be specified to extract corpus from a dataframe."
        source_values = np.array([])
        
        # e.g. conjoining test_order_name, test_result_name
        conjoined = tr.conjoin(df, cols=target_cols, remove_dup=remove_dup, transformed_vars_only=True, sep=" ")
        source_values = \
                    process_text(source_values=conjoined, col=col_new, 
                        clean=True, standardized=True, save=False, doc_type='training data')
    else:  
        # default to use LOINC field and MTRT as the source corpus
        print("(tfidf_pipeline) Use LOINC field and MTRT as the source corpus by default.")

        # B. Using LOINC LN and MTRT as corpus
        df_mtrt = kargs.get('df_mtrt', None)
        df_loinc = kargs.get('df_loinc', None)
        if df_mtrt is None: df_mtrt = LoincMTRT.load_table(dehyphenate=dehyphenate) 
        if df_loinc is None: df_loinc = loinc.load_loinc_table(dehyphenate=dehyphenate)

        # add external training data 
        # [todo]

        # [analysis]
        #    1. does PS appear in the long names? Yes
        ################################################################
        # df_mtrt_ps = df_mtrt[df_mtrt[col_mval].str.contains("--.*")]
        # codes_ps = df_mtrt_ps[col_mkey]
        # loinc_table = get_loinc_values(codes_ps, target_cols=[col_ln, col_sn, ], df_loinc=None, dehyphenate=True)

        # adict = {}
        # adict[col_mkey] = codes_ps
        # adict[col_ln] = loinc_table[col_ln]
        # adict[col_mval] = df_mtrt_ps[col_mval]
        # df_temp = DataFrame(adict) 
        # print("(parallel) Longnames for PS-present MTRTs:\n{}\n".format(df_temp.head(20)))
        ################################################################
        
        # -- load derived MTRT table
        # df_map = LoincMTRT.load_derived_loinc_to_mtrt()
        # columns: Test Result LOINC Code | Medivo Test Result Type | unit | compound | abbrev | note

        # now focus on loinc code and longname or MTRT strings only

        conjoined = lmt.get_corpora_from_merged_loinc_mtrt(dehyphenate=True, sep=" ", remove_dup=remove_dup)
        # ... target_cols <- LoincMTRT.cols_descriptor

        assert len(conjoined) == df_loinc.shape[0]

        source_values = process_text(source_values=conjoined, col=col_new, clean=True, standardized=True) 
        print("(model) Processed conjoined loinc LN and MTRT:\n{}\n".format(df_loinc_p.head(30)))


        # [analysis]
        ################################################################
        # ccmap = label_by_performance(cohort='hepatitis-c', categories=['easy', 'hard', 'low'])
        # codes_lsz = ccmap['low']
        # compare_longname_mtrt(df_mtrt=df_mtrt_p, df_loinc=df_loinc_p, codes=codes_lsz)
        ################################################################
        # ... for the most part, they are almost identical

    #######################################################
    # ... now we have the source corpus ready

    # preprocess_text_simple(source_value=source_values, value_default=value_default)
    print("... n={} source values".format( len(source_values) ))

    # model, mydict, corpus = build_tfidf_model(source_values=source_values)
    model = lm.build_tfidf_model(source_values=source_values, standardize=False)
    return model

def matching_score(query, code, model, loinc_lookup={}, standardize=False): 
    """
    Check in every LOINC document if query tokens exist and if the token exists, 
    then the tf-idf value is added to the matching score of that particular LOINC document

    Use
    ---
    1. query <- test_order_name

    """
    dehyphen = kargs.get('dehyphenate', True)
    remove_dup = kargs.get('remove_dup', False)

    if not loinc_lookup: 
        loinc_lookup = lmt.get_loinc_descriptors(dehyphenate=dehyphen, remove_dup=remove_dup)
    
    if standardize: 
        query = process_text(source_values=[query, ], clean=True, standardized=True, doc_type='query', verbose=0)[0]

    tokens = query.split()

    raise NotImplementedError("Coming soon :)")

def cosine_similarity(s1, s2, model, value_default=0.0):
    # from scipy.spatial import distance # cosine similarity
    v1 = model.transform([s1, ]).A.flatten()
    v2 = model.transform([s2, ]).A.flatten()
    s = 1.0-distance.cosine(v1, v2)

    if pd.isna(s): 
        # at least one of the inputs converts to a zero vector, which cannot be normalized, causing problem in computing cosine distance
        s = value_default
    
    # if s > 0: 
    #     print("(cosine_similarity) s1: {} => {}".format(s1, v1))
    #     print("(cosine_similarity) s2: {} => {}".format(s2, v2))
    return s

def compute_similarity_with_loinc(row, code, model, loinc_lookup={}, target_cols=[], value_default=0.0, **kargs):
    def iter_rules(target_cols, target_descriptors):
        if len(matching_rules) > 0: 
            for col, target_descriptors in matching_rules.items():
                for dpt in target_descriptors:
                    yield (col, dpt)
        else: 
            for col, dpt in itertools.product(target_cols, target_descriptors): 
                yield (col, dpt)

    #from scipy.spatial import distance # cosine similarity
    import itertools

    # --- LOINC table attributes
    col_ln, col_sn = LoincTable.long_name, LoincTable.short_name
    col_lkey = LoincTable.col_key
    col_com = LoincTable.col_com
    ##################################
    # --- Loinc to MTRT attributes (from Leela)
    col_mval = LoincMTRT.col_value
    col_mkey = LoincMTRT.col_key    # loinc codes in the mtrt table

    dehyphen = kargs.get('dehyphenate', True)
    remove_dup_tokens = kargs.get('remove_dup', False)
    matching_rules = kargs.get('matching_rules', {})  # a dictionary from T-attributes to Loinc descriptors
    class_label = kargs.get('label', '?') # optinal meta data for testing/interpretation 
    # return_name_values = kargs.get('return_name_values', False)
    
    target_descriptors = kargs.get('target_descriptors', [col_sn, col_ln, col_com])
    if len(target_cols) == 0: 
        target_cols = ['test_order_name', 'test_result_name', 'test_specimen_type', 'test_result_units_of_measure', ]
        # ... other attributes: 'panel_order_name'

        print("[feature generation] Variables defined wrt corpus from following attributes:\n{}\n".format(target_cols))
        assert np.all([col in row.index for col in target_cols])

    if not loinc_lookup: loinc_lookup = lmt.get_loinc_descriptors(dehyphenate=dehyphen, remove_dup=remove_dup_tokens, verify=True)
        
    # --- Matching rules 
    #     * compare {test_order_name, test_result_name} with SH, LN, Component
    scores = []
    attributes = [] 
    named_scores = defaultdict(dict)

    # for query, dpt in itertools.product(target_cols, target_descriptors):  
    for query, dpt in iter_rules(target_cols, target_descriptors):
        attributes.append(f"{query}_{dpt}")  # col, desc

        qv = row[query]
        try: 
            dv = loinc_lookup[code][dpt]
        except: 
            tval = code in loinc_lookup
            msg = "Code {} exists in the table? {}\n".format(code, tval)
            if tval: msg += "... table keys: {}\n".format( list(loinc_lookup[code].keys()) )
            raise ValueError(msg)

        qv = "" if pd.isna(qv) else process_string(qv, doc_type='query')
        dv = "" if pd.isna(dv) else process_string(dv, doc_type='doc')

        score = cosine_similarity(qv, dv, model)
        assert not pd.isna(score), "Null score | qv: {}, dv: {}".format(qv, dv)
        scores.append(score)
        named_scores[query][dpt] = score

    # if return_name_values: 
    #     return named_scores
    #     # return list(zip(attributes, scores))
    return scores, attributes, named_scores

def gen_sim_vars(df=None, target_cols=[], model=None, **kargs): 
    """
    Use a source corpus to train a TD-IDF model. Given this model
    convert target_cols (e.g. test_order_name) into vector forms and 
    then compute their similarities with respect to either LOINC LN or MTRT:  

    i) MTRT: obtained from leela 
    ii) LOINC Long Name (LN): obtained from LOINC table

    MTRT takes precedence over LOINC LN. 

    Params
    ------
    df_src: training data 
    df_loinc: loinc table 
    df_map: loinc to mtrt (e.g. Leela)

    Insights
    --------
    1. LOINC's long name just like MTRT can have [], (), -- special tokens

    Memo
    ----
    1. output_path = os.path.join(config.out_dir, "{}-sdist-vars.csv".format(dataType))

    2. Jaro Winkler distance function requires that both inputs be non-empty
    """
    from loinc import LoincTable, get_loinc_values, load_loinc_table, compare_longname_mtrt
    from loinc_mtrt import LoincMTRT
    from analyzer import label_by_performance   # analysis only
    import transformer as tr
    # from scipy.spatial import distance # cosine similarity
    from sklearn.metrics.pairwise import linear_kernel  

    verbose = kargs.get('verbose', 1)
    cohort = kargs.get('cohort', 'hepatitis-c')  # used to index into the desired dataset
    transformed_vars_only = kargs.get('transformed_vars_only', True)
    value_default = kargs.get('value_default', "")
    model = kargs.get('model', None)
    join_target_cols = kargs.get('join_target_cols', False)
    remove_dup = kargs.get('remove_dup', False)
    # col_conjoined = kargs.get('col_conjoined', tr.join_feature_names(target_cols, sep='_')) # name of the new column that combines 'target_cols'
    ##################################
    # --- training data attributes
    col_code = 'test_result_loinc_code'
    col_mtrt = 'medivo_test_result_type'
    ##################################
    # --- LOINC table attributes
    col_ln, col_sn = LoincTable.long_name, LoincTable.short_name
    col_lkey = LoincTable.col_key
    ##################################
    # --- Loinc to MTRT attributes (from Leela)
    col_mval = LoincMTRT.col_value
    col_mkey = LoincMTRT.col_key

    if len(train_cols) == 0: train_cols = [col_code, col_mtrt, ]

    # --- TF-IDF model
    if model is None: 
        source_values = kargs.get('source_values', [])
        #---------------------------------------------
        df_train = kargs.get('df_train', None)  # training corpus
        train_cols = kargs.get('train_cols', [])
        #---------------------------------------------
        # ... source_values takes precedence over df_train 

        assert (len(source_values) > 0) or (df_train is not None and 
            len(train_cols) > 0 and np.all([col in df_train.columns for col in train_cols]))

        # if df_train is None, then will default to use LOINC fields (longnames, shortnames) and MTRTs combined as a training corpus
        model = tfidf_pipeline(corpora=source_values, df_src=df_train, target_cols=train_cols)

    if df is None: 
        # load the original data, so that we can use the punctuation info to extract concepts (e.g. measurements are specified in brackets)
        isProcessed = False
        df = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=True, processed=isProcessed)

    # assert np.all([col in df.columns for col in target_cols]), "Missing some columns (any of {}) in the input".format(target_cols)
    ###########################################################################

    if not target_cols: target_cols = ['test_order_name', 'test_result_name', ] 

    if join_target_cols:
        col_conjoined = kargs.get('col_conjoined', tr.join_feature_names(target_cols, sep='_')) # name of the new column that combines 'target_cols'
        conjoined = tr.conjoin(df, cols=target_cols, remove_dup=remove_dup, transformed_vars_only=True, sep=" ")
        Xr = model.transform(conjoined)
        N = Xr.shape[0]
        assert df.shape[0] == N
        
        df = df.fillna("")
        
        # now compute cosine similarity wrt LOINC LN, and MTRT

        for target_col in ref_cols: 
            Xt = model.transform(df[target_col].values)
            assert Xr.shape[0] == Xt.shape[0]
            # for i in range(N):
            #     # r = 1.0-distance.cosine(Xr[i], Xt[i])
            #     r = linear_kernel(Xr[i], Xt[i]).flatten()
            #     sim_values.append(r)
            sim_values = np.diag(linear_kernel(Xr, Xt))


    # -- do pairwise comparison
    header = ['d_tfidf_loinc_to_mtrt', ] # 'd_jw_unit', 'd_jw_compound']
    # df['col_3'] = df.apply(lambda x: f(x.col_1, x.col_2), axis=1)

    if transformed_vars_only: 
        # columns(derived df_mtrt): Test Result LOINC Code, Medivo Test Result Type, unit, compound, abbrev, note
        df_output = DataFrame(columns=header)
    else: 
        df_incr = DataFrame(columns=header)
        df_output = pd.concat([df_src, df_incr], axis=1)
    col_key = kargs.get('col_key', LoincMTRT.col_key) # 'Test Result LOINC Code'

    # --- transform loinc field and MTRT field in the training data
    assert input_loinc.shape[0] == input_mtrt.shape[0]
    loinc_values = input_loinc[col_lkey].values
    X_loinc = model.transform(loinc_values)

    mtrt_values = input_mtrt[col_mval].values
    X_mtrt = model.transform(mtrt_values)

    return 

def predict_by_mtrt(mtrt_str='', target_code=None, df=None, **kargs):
    """

    Paramms
    -------
    code: if None, then predict a LOINC code 
          if a LOINC code is given, then predict it correctness by outputing a probability score


    Output
    ------
    a dictionary mapping from codes to probabilities

    Memo
    ----
    1. Case 1: MTRT is missing => infer from test_result_name, test_result_value and other attribtutes


       Case 2: MTRT is available => 
               compute (weighted) string distance, weighted by importance of tokens (e.g. td-idf scores)
    """
    col_key = kargs.get('col_key', LoincMTRT.col_key) # 'Test Result LOINC Code'
    if df is None: df = LoincMTRT.load_table(dehyphenate=True, dequote=True) # input_file/LoincMTRT.table
    
    # verify 
    for code in df['Test Result LOINC Code'].values: 
        assert code.find('-') < 0
    for v in df['Medivo Test Result Type'].values: 
        assert v.find('"') < 0
    assert len(mtrt_str) > 0 or code is not None

    
    print("(predict_by_mtrt) df.columns: {}".format(df.columns.values))
    print("(predict_by_mtrt) dim(df): {} | n_codes".format(df.shape, len(df[col_key].unique())) )
    
    # string matching algorithm
    df = extract_slots(df, save=True)
    # LoincMTRT.save_derived_loinc_to_mtrt(df)

    o = process_each('Albumin [Mass/volume] in Urine', code=17541)
    print(o)
    
    return

def encode_mtrt_tfidf(docs):  
    """

    Memo
    ----
    1. Medivo Test Result Type
    """

    return

def demo_read(**kargs):
    from gensim import corpora
    path_to_text_directory = "lsa_sports_food_docs"

    dictionary = corpora.Dictionary(ReadTxtFiles(path_to_text_directory))

    # Token to Id map
    dictionary.token2id
    # {'across': 0,
    #  'activity': 1,
    #  'although': 2,
    #  'and': 3,
    #  'are': 4,
    #  ...
    # } 
    return

def demo_parse(**kargs):
    """

    Related
    -------
    loinc.compare_longname_mtrt()

    """
    from analyzer import load_src_data, analyze_data_set, col_values, label_by_performance, col_values_by_codes, load_src_data
    from loinc import LoincTable, load_loinc_table
 
    cohort = "hepatitis-c"
    col_mtrt = 'medivo_test_result_type'
    col_loinc = 'test_result_loinc_code'

    col_key_loinc = LoincTable.col_code
    col_ln, col_sn = 'LONG_COMMON_NAME', 'SHORTNAME'

    df = analyze_data_set(**kargs)
    df_loinc = load_loinc_table(dehyphenate=True)
    # df = clean_mtrt(df, save=True)
    # ... MTRTs are standardized (punctuations removed)

    categories = ['easy', 'hard', 'low']  # low: low sample size
    ccmap = label_by_performance(cohort=cohort, categories=categories)

    codes_lsz = ccmap['low']
    adict = col_values_by_codes(codes_lsz, df=df, cols=[col_mtrt, col_loinc], mode='raw')

    for r, (target_code, mtrt_name) in enumerate(zip(adict[col_loinc], adict[col_mtrt])): 
        row_loinc = df_loinc.loc[df_loinc[col_key_loinc] == target_code]
        if not row_loinc.empty: 
            assert row_loinc.shape[0] == 1, "Found 1+ matches for code={}:\n{}\n".format(target_code, row_loinc[[col_key_loinc, col_ln]])

            long_name = row_loinc[col_ln].iloc[0] 
            short_name = row_loinc[col_sn].iloc[0]

            msg = "[{}] {}\n".format(r+1, target_code)
            msg += "    + MTRT: {}\n".format(mtrt_name)
            msg += "    + LN:   {}\n".format(long_name)
            print(msg)

    # values = col_values(df, col=col_target, n=10, mode='sampling', random_state=1)
    # print("(demo_parse) Sample values for col={}:\n{}\n".format(col_target, values))
    
    return

def demo_tfidf(**kargs): 
    from gensim.utils import simple_preprocess
    from smart_open import smart_open
    from gensim import models
    from gensim import corpora
    from pprint import pprint

    documents = ["This is the first line",
             "This is the second sentence",
             "This third document"]

    # Create the Dictionary and Corpus
    mydict = corpora.Dictionary([simple_preprocess(line) for line in documents])
    corpus = [mydict.doc2bow(simple_preprocess(line)) for line in documents]

    return

def feature_transform(df, target_cols=[], df_src=None, **kargs): 
    """
    Convert T-attributes into TF-IDF feature matrix with respect to the LOINC descriptors

    df -> X

    Params
    ------ 
    df: the data set containing the positive examples (with reliable LOINC assignments)

    """
    def show_evidence(row, code, code_neg=None, sdict={}, print_=False, min_score=0.0, label='?'):
        # sdict: T-attribute -> Loinc descriptor -> score
         
        code = row[LoincTSet.col_code]
        code_x = code   # Target code's corresponding T-attributes are to be matched against 

        msg = "(evidence) Found matching signals for code(+): {} (target aka \"reliable\" positive)\n".format(code)
        if code_neg is not None:
            msg = "(hypothesis) Found matching signals when {}(+) is REASSIGNED to {} (-)?\n".format(code, code_neg) 
            # ... the input code could be re-assigned to the code_neg
            label = '-'
            code_x = code_neg

        for col, entry in sdict.items(): 
            msg += "... {}: {}\n".format(col, row[col])  # a T-attribute and its value
            for col_loinc, score in entry.items():  # how does the current row's T attributes compared to the LOINC code's descriptors?
                if score > min_score: 
                    msg += "...  {}: {} => score: {} | {}({})\n".format(col_loinc, 
                        process_string(loinc_lookup[code_x][col_loinc]), score, code_x, label)
        if print_: print(msg)
        return msg

    from analyzer import load_src_data

    cohort = kargs.get('cohort', 'hepatitis-c')  # determines training data set
    target_codes = kargs.get('target_codes', []) 
    loinc_lookup = kargs.get('loinc_lookup', {})
    verify = kargs.get('verify', False)
    save = kargs.get('save', False)

    # --- LOINC table attributes
    col_ln, col_sn = LoincTable.long_name, LoincTable.short_name
    col_lkey = LoincTable.col_key
    col_com = LoincTable.col_com
    col_sys = LoincTable.col_sys
    col_method = LoincTable.col_method
    ######################################
    # --- training data attributes 
    col_target = LoincTSet.col_target # 'test_result_loinc_code'

    # use the entire source as training corpus

    # --- matching rules
    ######################################
    if len(target_cols) == 0: 
        print("[transform] By default, we'll compare following variables with LOINC descriptors:\n{}\n".format(target_cols))
        target_cols = ['test_order_name', 'test_result_name', 'test_result_units_of_measure', ]
    assert np.all(col in df.columns for col in target_cols)
    # ... other fields: 'panel_order_name'
    target_descriptors = [col_sn, col_ln, col_com, ]
    matching_rules = {'test_order_name': [col_sn, col_ln, col_com, ], 
                      'test_result_name': [col_sn, col_ln, col_com, ], 
                      # 'test_specimen_type': [col_sys, ], 
                      'test_result_units_of_measure': [col_sn, col_method], 
                      }
    ######################################
    highlight("Gathering training corpus (by default, use all data assoc. with target cohort: {} ...".format(cohort), symbol='#')
    if df_src is None: df_src = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=True, processed=True)
    col_new = 'corpus'
    # df_corpus = load_corpus(domain=cohort)
    df_corpus = get_corpora_by_loinc(df_src, target_cols, add_loinc_mtrt=True, 
                process_text=True, dehyphenate=True, verbose=1, return_dataframe=True, col_new=col_new, save=False)
    corpus = df_corpus[col_new].values
    assert len(corpus) == len(codeSet), "Each code is represented by one document!"
    # ------------------------------------
    highlight("Build TF-IDF model ...", symbol="#")
    model = build_tfidf_model(source_values=corpus, ngram_range=(1,3), lowercase=False, standardize=False, verify=True, max_features=50000)
    fset = model.get_feature_names()
    print("... TF-IDF model built | n(vars): {}".format(len(fset)))  # 11159
    ######################################

    non_codes = LoincTSet.null_codes # ['unknown', 'other', ]
    if len(target_codes) > 0: 
        # e.g. focus only on disease cohort-specific LOINC codes
        target_codes = list(set(target_codes) - set(non_codes))

        # select a subset of codes to create feature vectors
        dim0 = df.shape

        # df = df.loc[df[col_target].isin(target_codes)]
        df = loinc.select_samples_by_loinc(df, target_codes=target_codes, target_cols=target_cols, n_per_code=3) # opts: size_dict
        print("[transform] filtered input by target codes (n={}), dim(df):{} => {}".format(len(target_codes), dim0, df.shape))

    if not loinc_lookup: 
        loinc_lookup = lmt.get_loinc_descriptors(dehyphenate=True, remove_dup=False, recompute=True) # get_loinc_corpus_lookup_table(dehyphenate=True, remove_dup=False)
        print("[transform] size(loinc_lookup): {}".format(len(loinc_lookup)))

    # gen_sim_features(df_src=df_src, df_loinc=None, df_map=None, transformed_vars_only=True, verbose=1) 
    codes_missed = set([])
    n_codes = 0
    n_comparisons_pos = n_comparisons_neg = 0 
    n_detected = n_detected_in_negatives = 0
    pos_instances = []
    neg_instances = []
    N0 = df.shape[0]
    for code, dfc in df.groupby([LoincTSet.col_code, ]):
        n_codes += 1

        if n_codes % 10 == 0: print("[transform] Processing code #{}: {}  ...".format(n_codes, code))
        if code in LoincTSet.null_codes: continue
        if not code in loinc_lookup: 
            codes_missed.add(code)

        for r, row in dfc.iterrows():
            # ... remember that each LOINC may have n>1 instances but with different combinations of T-attributes
            
            if code in loinc_lookup: 
                # compute similarity scores between 'target_cols' and the LOINC descriptor of 'code' given trained 'model'
                sv, names, named_scores = \
                    compute_similarity_with_loinc(row, code, model=model, loinc_lookup=loinc_lookup, 
                        target_cols=target_cols, target_descriptors=target_descriptors) # target_descriptors
                pos_instances.append(sv)  # sv: a vector of similarity scores

                #########################################################################
                if verify: 
                    # positive_scores = defaultdict(dict)  # collection of positive sim scores, representing signals
                    tHasSignal = False
                    msg = f"[{r}] Code(+): {code}\n"
                    for target_col, entry in named_scores.items(): 
                        msg += "... Col: {}: {}\n".format(target_col, row[target_col])
                        msg += "... LN:  {}: {}\n".format(code, loinc_lookup[code][col_ln])
                        
                        for target_dpt, score in entry.items():
                            n_comparisons_pos += 1
                            if score > 0: 
                                n_detected += 1
                                msg += "    + {}: {}\n".format(target_dpt, score)
                                # nonzeros.append((target_col, target_dpt, score))
                                # positive_scores[target_col][target_dpt] = score
                                tHasSignal = True
                    # ------------------------------------------------
                    if not tHasSignal: msg += "    + No similar properties found between {} and {} #\n".format(target_cols, target_descriptors)
                    print(msg)
                    if tHasSignal: 
                        highlight(show_evidence(row, code=code, sdict=named_scores, print_=False), symbol='#')
                #########################################################################

                # [Q] what happens if we were to assign an incorrect LOINC code, will T-attributes stay consistent with its LOINC descriptor? 
                codes_negative = loinc.sample_negatives(code, target_codes, n_samples=10, model=None, verbose=1)
                
                for code_neg in codes_negative: 

                    if code_neg in loinc_lookup: 
                        sv, names, named_scores = \
                            compute_similarity_with_loinc(row, code_neg, model=model, loinc_lookup=loinc_lookup, 
                                target_cols=target_cols, target_descriptors=target_descriptors) # target_descriptors
                        neg_instances.append(sv)  # sv: a vector of similarity scores
                        
                        # ------------------------------------------------
                        if verify: 
                            tHasSignal = False
                            # positive_scores = defaultdict(dict)
                            msg = f"[{r}] Code(-): {code_neg}\n"
                            for target_col, entry in named_scores.items(): 
                                msg += "... Col: {}: {}\n".format(target_col, row[target_col])
                                msg += "... LN:  {}: {}\n".format(code_neg, loinc_lookup[code_neg][col_ln])

                                # nonzeros = []
                                for target_dpt, score in entry.items():
                                    n_comparisons_neg += 1
                                    if score > 0: 
                                        n_detected_in_negatives += 1
                                        msg += "    + {}: {}\n".format(target_dpt, score)
                                        # positive_scores[target_col][target_dpt] = score
                                        tHasSignal = True

                                if tHasSignal: 
                                    msg += "    + Found similar properties between T-attributes(code={}) and negative: {}  ###\n".format(code, code_neg)
                                print(msg)  
                                if tHasSignal: 
                                    highlight(show_evidence(row, code=code, code_neg=code_neg, sdict=positive_scores, print_=False), symbol='#')
                        # ------------------------------------------------
    X = np.vstack([pos_instances, neg_instances])
    print("[transform] from n(df)={}, we created n={} training instances".format(N0, X.shape[0]))
    if save: 
        pass  

    # note:        
    return X 

def demo_create_vars(**kargs):
    def save_corpus(df, domain, output_dir='data', output_file=''): 
        if not output_file: output_file = f"{domain}.corpus"
        output_path = os.path.join(output_dir, output_file)
        print("(demo) Saving corpora_by_loinc output to:\n{}\n".format(output_path))
        df.to_csv(output_path, index=False, header=True)
        return 
    def load_corpus(domain, input_dir='data', input_file=""):
        if not input_file: input_file = f"{domain}.corpus"
        input_path = os.path.join(input_dir, input_file)
        print("(demo) Loading corpora_by_loinc doc from:\n{}\n".format(input_path))
        df =  None
        if os.path.exists(input_path): 
            df = pd.read_csv(input_path, sep=",", header=0, index_col=None, error_bad_lines=False)
        else: 
            print("... No doc found!")
        return df
    def show_evidence(row, code, code_neg=None, sdict={}, print_=False, min_score=0.0, label='?'):
        # sdict: T-attribute -> Loinc descriptor -> score
         
        code = row[LoincTSet.col_code]
        code_x = code   # Target code's corresponding T-attributes are to be matched against 

        msg = "(evidence) Found matching signals for code(+): {} (target aka \"reliable\" positive)\n".format(code)
        if code_neg is not None:
            msg = "(hypothesis) Found matching signals when {}(+) is REASSIGNED to {} (-)?\n".format(code, code_neg) 
            # ... the input code could be re-assigned to the code_neg
            label = '-'
            code_x = code_neg

        for col, entry in sdict.items(): 
            msg += "... {}: {}\n".format(col, row[col])  # a T-attribute and its value
            for col_loinc, score in entry.items():  # how does the current row's T attributes compared to the LOINC code's descriptors?
                if score > min_score: 
                    msg += "...  {}: {} => score: {} | {}({})\n".format(col_loinc, 
                        process_string(loinc_lookup[code_x][col_loinc]), score, code_x, label)
        if print_: print(msg)
        return msg

    from analyzer import label_by_performance, col_values_by_codes, load_src_data
    from feature_analyzer import plot_heatmap  

    # --- LOINC table attributes
    col_ln, col_sn = LoincTable.long_name, LoincTable.short_name
    col_lkey = LoincTable.col_key
    col_com = LoincTable.col_com
    col_sys = LoincTable.col_sys
    col_method = LoincTable.col_method
    ######################################

    # --- Cohort definition (based on target condition and classifier array performace)
    ######################################
    cohort = "hepatitis-c"
    col_target = 'test_result_loinc_code'
    categories = ['easy', 'hard', 'low']  # low: low sample size

    ccmap = label_by_performance(cohort='hepatitis-c', categories=categories)

    codes_lsz = ccmap['low']
    print("(demo) n_codes(low sample size): {}".format(len(codes_lsz)))
    codes_hard = ccmap['hard']
    print("...    n_codes(hard): {}".format(len(codes_hard)))
    target_codes = list(set(np.hstack([codes_hard, codes_lsz])))

    # remove non-codes 
    non_codes = ['unknown', 'other', ]
    target_codes = list(set(target_codes) - set(non_codes))

    df_src = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=True, processed=True)
    codeSet = df_src[col_target].unique()
    ######################################

    # use the entire source as training corpus

    # --- matching rules
    ######################################
    target_cols = ['test_order_name', 'test_result_name', ] # 'test_result_units_of_measure'
    # ... other fields: 'panel_order_name'

    target_descriptors = [col_sn, col_ln, col_com, ]
    matching_rules = {'test_order_name': [col_sn, col_ln, col_com, ], 
                      'test_result_name': [col_sn, col_ln, col_com, ], 
                      # 'test_specimen_type': [col_sys, ], 
                      'test_result_units_of_measure': [col_sn, col_method]
                      }
    ######################################
    
    highlight("Constructing corpus ...", symbol='#')
    col_new = 'corpus'
    df_corpus = load_corpus(domain=cohort)
    if df_corpus is None: 
        df_corpus = get_corpora_by_loinc(df_src, target_cols, add_loinc_mtrt=True, 
            process_text=True, dehyphenate=True, verbose=1, return_dataframe=True, col_new=col_new, save=False)
        save_corpus(df_corpus, domain=cohort)
    corpus = df_corpus[col_new].values
    assert len(corpus) == len(codeSet), "Each code is represented by one document!"
    ######################################

    highlight("Build TF-IDF model ...", symbol="#")
    model = build_tfidf_model(source_values=corpus, ngram_range=(1,3), lowercase=False, standardize=False, verify=True, max_features=50000)
    fset = model.get_feature_names()
    print("... TF-IDF model built | n(vars): {}".format(len(fset)))  # 11159
    ######################################

    # adict = col_values_by_codes(target_codes, df=df_src, cols=['test_result_name', 'test_order_name'], mode='raw')
    df_src = df_src.loc[df_src[col_target].isin(target_codes)]
    print("(demo) dim(input): {}".format(df_src.shape))

    loinc_lookup = lmt.get_loinc_descriptors(dehyphenate=True, remove_dup=False, recompute=True) # get_loinc_corpus_lookup_table(dehyphenate=True, remove_dup=False)
    print("(demo) size(loinc_lookup): {}".format(len(loinc_lookup)))

    # gen_sim_features(df_src=df_src, df_loinc=None, df_map=None, transformed_vars_only=True, verbose=1) 
    codes_missed = set([])
    n_codes = 0
    n_comparisons_pos = n_comparisons_neg = 0 
    n_detected = n_detected_in_negatives = 0
    pos_instances = []
    neg_instances = []
    attributes = []
    for code, dfc in df_src.groupby([LoincTSet.col_code, ]):
        n_codes += 1
        if code == LoincMTRT.col_unknown: continue
        if not code in loinc_lookup: 
            codes_missed.add(code)

        for r, row in dfc.iterrows():
            # ... don't deal with LOINC of the unknown/other category
            
            if code in loinc_lookup: 
                # compute similarity scores between 'target_cols' and the LOINC descriptor of 'code' given trained 'model'
                scores, attributes, named_scores = \
                    compute_similarity_with_loinc(row, code, model=model, loinc_lookup=loinc_lookup, 
                        target_cols=target_cols, target_descriptors=target_descriptors) # target_descriptors
                pos_instances.append(scores)

                # ------------------------------------------------
                positive_scores = defaultdict(dict)
                msg = f"[{r}] Code(+): {code}\n"
                for target_col, entry in named_scores.items(): 
                    msg += "... Col: {}: {}\n".format(target_col, row[target_col])
                    msg += "... LN:  {}: {}\n".format(code, loinc_lookup[code][col_ln])
                    
                    for target_dpt, score in entry.items():
                        n_comparisons_pos += 1
                        if score > 0: 
                            n_detected += 1
                            msg += "    + {}: {}\n".format(target_dpt, score)
                            # nonzeros.append((target_col, target_dpt, score))
                            positive_scores[target_col][target_dpt] = score
                # ------------------------------------------------
                if len(positive_scores) == 0: 
                    msg += "    + No similar properties found between {} and {} #\n".format(target_cols, target_descriptors)
                    print(msg)
                if len(positive_scores) > 0: 
                    highlight(show_evidence(row, code=code, sdict=positive_scores, print_=False), symbol='#')

                #########################################################################
                highlight("What if we assign a wrong code deliberately?", symbol='#')
                # [Q] what happens if we were to assign an incorrect LOINC code, will T-attributes stay consistent with its LOINC descriptor? 
                codes_negative = loinc.sample_negatives(code, target_codes, n_samples=10, model=None, verbose=1)
                tFoundMatchInNeg = False
                for code_neg in codes_negative: 

                    if code_neg in loinc_lookup: 
                        scores, attributes, named_scores = \
                            compute_similarity_with_loinc(row, code_neg, model=model, loinc_lookup=loinc_lookup, 
                                target_cols=target_cols, target_descriptors=target_descriptors) # target_descriptors
                        neg_instances.append(scores)
                        
                        positive_scores = defaultdict(dict)
                        msg = f"[{r}] Code(-): {code_neg}\n"
                        for target_col, entry in named_scores.items(): 
                            msg += "... Col: {}: {}\n".format(target_col, row[target_col])
                            msg += "... LN:  {}: {}\n".format(code_neg, loinc_lookup[code_neg][col_ln])

                            # nonzeros = []
                            for target_dpt, score in entry.items():
                                n_comparisons_neg += 1
                                if score > 0: 
                                    n_detected_in_negatives += 1
                                    msg += "    + {}: {}\n".format(target_dpt, score)
                                    positive_scores[target_col][target_dpt] = score

                            if len(positive_scores) > 0: 
                                msg += "    + Found similar properties between T-attributes(code={}) and negative: {}  ###\n".format(code, code_neg)
                                print(msg)  
                            if len(positive_scores) > 0: 
                                tFoundMatchInNeg = True
                                highlight(show_evidence(row, code=code, code_neg=code_neg, sdict=positive_scores, print_=False), symbol='#')
                if tFoundMatchInNeg: 
                    n_detected_in_negatives += 1

    print("... There are n={} codes not found on the LONIC+MTRT corpus table:\n{}\n".format(len(codes_missed), codes_missed))
    r_detected = n_detected/(n_comparisons_pos+0.0)
    r_detected_in_neg = n_detected_in_negatives/(n_comparisons_neg+0.0)
    print("...... Among N={} codes, r(detected): {}, r(detected in any -): {} | method=\"tfidf\"".format(n_codes, r_detected, r_detected_in_neg))
    
    # --- Visualize
    col_label = MatchmakerFeatureSet.col_target  # 'label'
    df_pos = DataFrame(pos_instances, columns=attributes)
    df_pos[col_label] = 1
    df_neg = DataFrame(neg_instances, columns=attributes)
    df_neg[col_label] = 0
    # X = np.vstack([pos_instances, neg_instances])
    # y = np.vstack([np.repeat(1, len(pos_instances)), np.repeat(0, len(neg_instances))])

    n_display = 10
    vtype = subject = 'tfidf'

    df_match = pd.concat([df_pos, df_neg], ignore_index=True)

    parentdir = os.path.dirname(os.getcwd())
    testdir = os.path.join(parentdir, 'test')  # e.g. /Users/<user>/work/data
    output_file = f'{vtype}-vars.csv'
    output_path = os.path.join(testdir, output_file)

    # Output
    # --------------------------------------------------------
    df_match.to_csv(output_path, index=False, header=True)
    # --------------------------------------------------------

    tabulate(df_match.sample(n=n_display), headers='keys', tablefmt='psql')

    # ... tif may not be supported (Format 'tif' is not supported (supported formats: eps, pdf, pgf, png, ps, raw, rgba, svg, svgz))
    # plot_heatmap(data=df_match, output_path=fpath)

    # --- matching scores 
    # matching_score()

    return df_match

def demo_create_vars_part2(**kargs): 
    """

    Memo
    ----
   
    1. distance metric (correlation, cocine, euclidean, ...)

       https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    2. linkage method 

       https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

       + distance between two clusetrs d(s, t)
       + a distance matrix is maintained at earch iteration

    3. In cluster analysis (and PCA), var standardization seems better 
       https://sebastianraschka.com/Articles/2014_about_feature_scaling.html#z-score-standardization-or-min-max-scaling

       + clustermap, running hierarchical clustering over heatmap 
         https://seaborn.pydata.org/generated/seaborn.clustermap.html

    4. coloring 

       + heatmap: https://seaborn.pydata.org/generated/seaborn.heatmap.html
       + color palettes: https://seaborn.pydata.org/tutorial/color_palettes.html

       + color picker
         https://htmlcolorcodes.com/color-picker/

    5. Axis labels 
       https://www.drawingfromdata.com/how-to-rotate-axis-labels-in-seaborn-and-matplotlib

       To rotate labels:
           need to reference the Axes from the underlying Heatmap and rotate these

           e.g. cg = sns.clustermap(df, metric="correlation")
                plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

    """
    def relabel(df, target_cols=[]):
        if not target_cols: target_cols = df.columns

        # remove prefix
        new_cols = []
        for i, col in enumerate(target_cols): 
            new_cols.append(col.replace("test_", ""))
        target_cols = new_cols

        new_cols = []
        loinc_abbrev = LoincTable.cols_abbrev 
        for i, col in enumerate(target_cols):
            tFoundMatch = False
            for name, abbrev in loinc_abbrev.items(): 
                if col.find(name) >= 0: 
                    new_cols.append(col.replace(name, abbrev))
                    print("... {} -> {}".format(col, col.replace(name, abbrev)))
                    tFoundMatch = True
                    break
            if not tFoundMatch: 
                new_cols.append(col) # new column is the old column
        target_cols = new_cols
        
        return df.rename(columns=dict(zip(df.columns.values, target_cols)))

    import seaborn as sns
    import matplotlib
    matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
    import matplotlib.pyplot as plt

    # select plotting style; must be called prior to pyplot
    plt.style.use('seaborn')  # values: {'seaborn', 'ggplot', }  
    from utils_plot import saveFig
    # from sklearn import preprocessing
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from data_processor import toXY, down_sample
    from feature_analyzer import plot_heatmap, plot_data_matrix

    plt.clf()
    
    # Output parameters
    cohort = kargs.get('cohort', 'hepatitis-c')
    col_label = kargs.get('label', MatchmakerFeatureSet.col_target)

    # Data parameters
    tScale = False  # set zcore = 1 within clustermap() instead
    tPerturb = False

    # ---------------------------------------------
    # sns.set_color_codes("pastel")
    # sns.set(color_codes=True)
    n_colors = 256 # Use 256 colors for the diverging color palette
    # palette = sns.diverging_palette(20, 220, n=n_colors) # Create the palette
    palette = sns.color_palette("coolwarm", 10)
    # sns.palplot(sns.color_palette("coolwarm", 7))
    # sns.palplot(palette)
    # ---------------------------------------------
    n_display = 10
    vtype = subject = kargs.get('vtype', 'tfidf')

    cols_y = [col_label, ]
    cols_untracked = []
    # ---------------------------------------------

    # read the feature vectors
    df_match = kargs.get('df_match', None)
    if df_match is None: 
        parentdir = os.path.dirname(os.getcwd())
        testdir = os.path.join(parentdir, 'test')  # e.g. /Users/<user>/work/data
        input_file = kargs.get("input_file", f'{vtype}-vars.csv') 
        input_path = os.path.join(testdir, input_file)
        assert os.path.join(input_path), "Invalid data file:\n{}\n".format(input_path)
        df_match = pd.read_csv(input_path, sep=",", header=0, index_col=None, error_bad_lines=False)
    # ---------------------------------------------

    # limit sample size to unclutter plot 
    n_samples = 50
    df_match = down_sample(df_match, col_label=col_label, n_samples=n_samples)
    df_match = df_match.sort_values(by=[col_label, ], ascending=True)
    df_match = relabel(df_match)
    # ---------------------------------------------

    df_pos = df_match[df_match[col_label]==1]
    df_neg = df_match[df_match[col_label]==0]

    # -- perturb the negative examples by a small random noise
    # ep = np.min(df_pos.values[df_pos.values > 0])
    # df_neg += np.random.uniform(ep/100, ep/10, df_neg.shape)
    # print("... ep: {} => perturbed df_neg:\n{}\n".format(ep, df_neg.head(10)))
    # df_match = pd.concat([df_pos, df_neg])

    # ---------------------------------------------
    # labels = df_match.pop(col_label)  # this is an 'inplace' operation
    labels = df_match[col_label]  # labels: a Series
    n_labels = np.unique(labels.values)
    # ---------------------------------------------

    lut = {0: "#3933FF", 1: "#FF3368"} # dict(zip(labels.unique(), "rb"))
    # negative (blue): #3933FF, #3358FF, #e74c3c
    # positive (red) : #FF3368,  #3498db 
    print("... lut: {}".format(lut))
    row_colors = labels.map(lut)

    # detect zero vectors
    df_zeros = df_match.loc[(df_match.T == 0).all()]
    print("... found n={} zero vectors (which cannot be normalized; troubled in computing cosine, correlation)".format(df_zeros.shape[0]))
    df_pos_zeros = df_pos.loc[(df_pos.T == 0).all()]
    df_neg_zeros = df_neg.loc[(df_neg.T == 0).all()]
    print("... n(pos): {}, n(pos, 0): {}, ratio: {}".format(df_pos.shape[0], df_pos_zeros.shape[0], df_pos_zeros.shape[0]/df_pos.shape[0]))
    print("... n(neg): {}, n(neg, 0): {}, ratio: {}".format(df_neg.shape[0], df_neg_zeros.shape[0], df_neg_zeros.shape[0]/df_neg.shape[0]))

    # standardize the data
    print("... col(df_match): {}".format(df_match.columns.values))

    if tScale:
        X, y, fset, lset = toXY(df_match, cols_y=cols_y, scaler='standardize', perturb=tPerturb)
        print("... feature set: {}".format(fset))

        # df_match = DataFrame(np.hstack([X, y, z]), columns=fset+lset+cols_untracked)
        dfX = DataFrame(X, columns=fset)
        # ... don't include y here

        df_zeros = dfX.loc[(dfX.T == 0).all()]
        print("... After standardization, found n={} zero vectors".format(df_zeros.shape[0]))
    else: 
        dfX = df_match.drop(cols_y, axis=1)

    highlight("(demo) 1. Plotting dendrogram (on top of heatmap) ...", symbol='#')

    # Normalize the data within the rows via z_score
    sys.setrecursionlimit(10000)
    g = sns.clustermap(dfX, figsize=(15, 24.27), row_colors=row_colors, z_score=1,
                cmap='vlag', metric='cosine', method='complete')  # z_score=1, vmin=0, vmax=1
    # ... need to pass dataframe dfX instead of X! 

    # g.set_xticklabels(g.get_xticklabels(), rotation=45)
    # print(dir(g.ax_heatmap))
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=45)
    # plt.setp(g.ax_heatmap.secondary_yaxis(), rotation=45)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=16, fontweight='light')
    # ... horizontalalignment='right' shifts labels to the left
    
    # colors:  cmap="vlag", cmap="mako", cmap=palette
    # normalization: z_score, stndardize_scale

    output_path = os.path.join(testdir, f'clustermap-{vtype}-{cohort}.pdf')
    # g.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight') 
    saveFig(plt, output_path, dpi=300)

    ################################################
    df_match = pd.read_csv(input_path, sep=",", header=0, index_col=None, error_bad_lines=False)
    n_samples = 50
    df_match = down_sample(df_match, col_label=col_label, n_samples=n_samples)
    df_match = relabel(df_match)
    df_pos = df_match[df_match[col_label]==1]
    df_neg = df_match[df_match[col_label]==0]
    # ... no X scaling

    # --- Enhanced heatmap 
    highlight("(demo) 2. Visualize feature values > (+) examples should higher feature values, while (-) have low to zero values")
    output_path = os.path.join(testdir, f'{vtype}-pos-match-{cohort}.png')
    plot_data_matrix(df_pos, output_path=output_path, dpi=300)

    output_path = os.path.join(testdir, f'{vtype}-neg-match-{cohort}.png')
    plot_data_matrix(df_neg, output_path=output_path, dpi=300)

    #################################################
    # --- PCA plot 

    plt.clf()
    highlight("(demo) 3. PCA plot", symbol='#')
    
    # X = df_match.drop(cols_y+untracked, axis=1).values

    # read in the data again 
    df_match = pd.read_csv(input_path, sep=",", header=0, index_col=None, error_bad_lines=False)
    df_match = down_sample(df_match, col_label=col_label)
    X, y, fset, lset = toXY(df_match, cols_y=cols_y, scaler='standardize')

    # print("... dim(X): {} => X=\n{}\n".format(X.shape, X))
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    df_match['pca1'] = pca_result[:,0]
    df_match['pca2'] = pca_result[:,1] 

    np.random.seed(42)
    rndperm = np.random.permutation(df_match.shape[0])

    n_colors = 2  # len(labels)
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="pca1", y="pca2",
        hue="label",
        palette=sns.color_palette("hls", n_colors),
        data=df_match.loc[rndperm,:],   # rndperm is redundant here
        legend="full",
        alpha=0.3
    )

    output_path = os.path.join(testdir, f'PCA-{vtype}-{cohort}.png') 
    saveFig(plt, output_path, dpi=300)

    #################################################
    # --- tSNE Plot

    highlight("(demo) 4. t-SNE plot", symbol='#')
    plt.clf()
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=15, n_iter=300)
    tsne_results = tsne.fit_transform(X)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df_match['tsne1'] = tsne_results[:,0]
    df_match['tsne2'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))

    # note: requires seaborn-0.9.0
    sns.scatterplot(
        x="tsne1", y="tsne2",
        hue="label",
        palette=sns.color_palette("hls", n_colors),
        data=df_match,
        legend="full",
        alpha=0.3
    )

    output_path = os.path.join(testdir, f'tSNE-{vtype}-{cohort}.png') 
    saveFig(plt, output_path, dpi=300)
        
    return

def demo_experta(**kargs):
    from random import choice
    # from experta import *

    return

def demo_loinc_mtrt(): 
    from analyzer import compare_col_values, load_src_data
    from loinc import expand_by_longname

    cohort = "hepatitis-c"

    df = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=True, processed=True)
    dim0 = df.shape
    df = expand_by_longname(df, col_src='test_result_loinc_code', 
           col_derived='test_result_loinc_longname', df_ref=None, transformed_vars_only=False, dehyphenate=True)
    print("(demo) dim0: {} => dim(df): {}, col(df):\n{}\n".format(dim0, df.shape, df.columns.values))

    cols = ['test_result_loinc_longname', 'medivo_test_result_type',]
    compare_col_values(df, cols=cols, n=10, mode='sampling', verbose=1, random_state=53)

    return

def demo_corpus(mode='code'):
    from analyzer import compare_col_values, load_src_data
    cohort = "hepatitis-c"
    df = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=True, processed=True)

    # select n rows for each code (e.g. n=1)
    target_cols = ['test_order_name', 'test_result_name', 'test_specimen_type', 'panel_order_name', 'test_result_units_of_measure', ]
    
    if mode.startswith('reg'): 
        corpora = get_corpora_from_dataframe(df, target_cols, add_loinc_mtrt=True, processed=True, dehyphenate=True)
        col_corpus = 'corpus'
        df[col_corpus] = corpora
        highlight("(demo) n(docs): {}".format(df.shape[0]))
        compare_col_values(df, target_cols+[col_corpus, ], n=50, mode='sampling', include_6parts=True, include_mtrt=True, verbose=1, random_state=53)
    else: 
        highlight("(demo) Now make corpora by LOINC code such that each code has its down document.")
        return_dataframe = True  # for debugging
        col_corpus = 'corpus'
        df_corpus = get_corpora_by_loinc(df, target_cols, 
            add_loinc_mtrt=True, process_text=True, dehyphenate=True, verbose=1, 
            return_dataframe=return_dataframe, col_new=col_corpus)
        highlight("(demo) n(docs): {}".format(df_corpus.shape[0]))
        assert df_corpus.shape[0] == len(df[LoincTSet.col_target].unique())
        compare_col_values(df_corpus, target_cols+[col_corpus, ], n=50, mode='sampling', include_6parts=True, include_mtrt=True, verbose=1, random_state=53)
    
    return

def demo_predict(**kargs): 

    # predict_by_mtrt()

    # compute_similarity_with_loinc(row, code, model=None, corpus=None, loinc_lookup={}, target_cols=[], value_default=0.0, **kargs)

    return

def test(**kargs): 

    ### Parsing, cleaing, standardizing 
    # demo_parse()
    # demo_loinc_mtrt()

    # --- Training Corpus 
    demo_corpus(mode='reg')

    #--- Text feature generation
    demo_create_vars()
    demo_create_vars_part2()

    #--- Basic LOINC Prediction
    # demo_predict()

    return

if __name__ == "__main__":
    test()