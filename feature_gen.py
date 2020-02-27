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
from loinc import LoincTable, LoincTSet
from loinc_mtrt import LoincMTRT
import loinc_mtrt as lmt

from utils_sys import highlight
import language_model as langm
# from language_model import build_tfidf_model
import config

import common, text_processor
from text_processor import process_text
from CleanTextData import standardize 

import feature_gen_tfidf as fg_tfidf
import feature_gen_sdist as fg_sdist

def build_tfidf_model(cohort='hepatitis-c', df_src=None, target_cols=[], **kargs):
    """
    Input
    -----
    cohort: Use 'cohort' to index into the training data

    """
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
            print("[load] Could not load LOINC document; recompute a new corpus ...")
        return df

    from analyzer import load_src_data

    ngram_range = kargs.get('ngram_range', (1,3))
    max_features = kargs.get('max_features', 50000)
    col_new = kargs.get('col_new', 'corpus')
    verbose = kargs.get('verbose', 1)
    col_target = LoincTSet.col_target

    if len(target_cols) == 0:
        target_cols = ['test_order_name', 'test_result_name', 'test_result_units_of_measure', ]

    # ------------------------------------
    if df_src is None: df_src = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=True, processed=True)

    df_corpus = load_corpus(domain=cohort)
    if df_corpus is None: 
        df_corpus = fg_tfidf.get_corpora_by_loinc(df_src, target_cols, add_loinc_mtrt=True, 
            process_text=True, dehyphenate=True, verbose=1, return_dataframe=True, col_new=col_new, save=False)
        save_corpus(df_corpus, domain=cohort)
    corpus = df_corpus[col_new].values
    # ------------------------------------
    codeSet = df_src[col_target].unique()
    assert len(corpus) == len(codeSet), "Each code is represented by one document!"
    # ------------------------------------
    if verbose: highlight("Build TF-IDF model ...", symbol="#")
    model = langm.build_tfidf_model(source_values=corpus, ngram_range=ngram_range,
                     lowercase=False, standardize=False, verify=verbose, max_features=max_features)
    if verbose: 
        fset = model.get_feature_names()
        print("... TF-IDF model built | n(vars): {}".format(len(fset)))  # 11159

    return model

def feature_transform(df, **kargs): 
    """
    Convert T-attributes into the following sets of features
      1) Similarity scores between TF-IDF transformed T-attributes and the LOINC descriptors
      2) String distance-based similarity scores between T-attributes and the LOINC descriptors
      3) other features

    df -> X

    Input
    ----- 
    df: the data set containing the positive examples (with reliable LOINC assignments)

    target_cols: When building TF-IDF (or other document embedding models), gather the text data 
                 only from these columns

    target_codes: if given, focus on rows documented with these LOINC codes

    loinc_lookup
    vars_lookup

    Use 
    --- 
    1. Generate training examples (correct assignments vs incorrect assignments)
    2. Transform new instances for prediction, in which case, we do not need to generate 
       negative examples

    """
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
    def show_evidence(row, code_neg=None, sdict={}, print_=False, min_score=0.0):
        # sdict: T-attribute -> Loinc descriptor -> score

        code = row[LoincTSet.col_code]
        msg = "(show_evidence) code: {}\n".format(code)
        if code_neg is not None:
            msg = "(show_evidence) {} ->? {}\n".format(code, code_neg) 
            # ... the input code could be re-assigned to the code_neg

        for col, entry in sdict.items(): 
            msg += "... {}: {} ~ \n".format(col, row[col])  # a T-attribute and its value
            for col_loinc, score in entry.items():
                if score > min_score: 
                    msg += "    + {}: {} => score: {}\n".format(col_loinc, loinc_lookup[code][col_loinc], score)
        if print_: print(msg)
        return msg

    from analyzer import load_src_data

    # --- LOINC table attributes
    col_ln, col_sn = LoincTable.long_name, LoincTable.short_name
    col_lkey = LoincTable.col_key
    col_com = LoincTable.col_com
    col_sys = LoincTable.col_sys
    col_method = LoincTable.col_method
    col_prop = LoincTable.col_prop
    # --- training data attributes 
    col_target = LoincTSet.col_target # 'test_result_loinc_code'

    cohort = kargs.get('cohort', '')  # determines training data set
    df_src = kargs.get('df_src', None)
    if not cohort: assert df_src is not None

    target_cols = kargs.get('target_cols', ['test_order_name', 'test_result_name', 'test_result_units_of_measure', ])
    target_codes = kargs.get('target_codes', list(df[col_target].unique())) 

    loinc_lookup = kargs.get('loinc_lookup', {})
    vars_lookup = kargs.get('vars_lookup', {})
    verify = kargs.get('verify', True)
    save = kargs.get('save', False)

    # --- TF-IDF specific 
    ngram_range = kargs.get('ngram_range', (1,3))
    max_features = kargs.get('max_features', 50000)

    # --- Data specific 
    tGenNegative = kargs.get('gen_negative', True)
    n_per_code = kargs.get("n_per_code", 3)
        
    # --- Define matching rules
    ######################################
    print("[transform] The following T-attributes are to be compared with LOINC descriptors:\n{}\n".format(target_cols))
    assert np.all(col in df.columns for col in target_cols)

    # ... other fields: 'panel_order_name'
    target_descriptors = [col_sn, col_ln, col_com, col_sys, ]
     
    matching_rules = kargs.get('matching_rules', {})
    if not matching_rules: 
        matching_rules = {target_col: target_descriptors for target_col in target_cols}
        
        # e.g.
        # matching_rules = {'test_order_name': [col_sn, col_ln, col_com, col_sys, ], 
        #                   'test_result_name': [col_sn, col_ln, col_com, col_sys, col_prop, ], 
        #                   'test_specimen_type': [col_sys, ], 
        #                   'test_result_units_of_measure': [col_sn, col_prop], 
        #                   }
    ######################################
    # default to use all LOINC codes associatied with the given cohort as the source of training corpus
    model = build_tfidf_model(cohort=cohort, df_src=df_src, target_cols=target_cols, ngram_range=ngram_range, max_features=max_features)
    # note: if df_src is given, cohort is ignored

    # note that df_src is the source used to generate the corpus
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
        df = loinc.select_samples_by_loinc(df, target_codes=target_codes, target_cols=target_cols, n_per_code=n_per_code) # opts: size_dict
        print("[transform] filtered input by target codes (n={}), dim(df):{} => {}".format(len(target_codes), dim0, df.shape))

    # load LOINC descriptors
    if not loinc_lookup: 
        # LOINC descriptors are generated via consolidating LOINC and MTRT tables that involve the following operations: 
        # ... 1) merge 2) conjoin (see transformer.conjoin())
        loinc_lookup = lmt.get_loinc_descriptors(dehyphenate=True, remove_dup=False, recompute=True) # get_loinc_corpus_lookup_table(dehyphenate=True, remove_dup=False)
        print("[transform] LOINC descriptors dict | size(loinc_lookup): {}".format(len(loinc_lookup)))

    # Load transformed variables generated by string distance-based approach (see feature_gen_sdist)
    if not vars_lookup: 
       # Variables derived via string distance measures are generated via demo_create_vars_init() (to be "formalized")
       vars_lookup = LoincTSet.load_sdist_var_descriptors(target_cols)
       assert len(vars_lookup) == len(target_cols)
       print("[transform] String-distance variables | size(vars_lookup): {}".format(len(vars_lookup)))

    codes_missed = set([])
    n_codes = 0
    n_comparisons_pos = n_comparisons_neg = 0 
    n_detected = n_detected_in_negatives = 0
    pos_instances = []
    neg_instances = []
    attributes = []
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

                # --- String distance model
                sv1, attr1, named_scores1 = \
                    fg_sdist.compute_similarity_with_loinc(row, code, loinc_lookup=loinc_lookup, vars_lookup=vars_lookup,
                        matching_rules=matching_rules, # this takes precedence over product(target_cols, target_descriptors)
                            add_sdist_vars=False,

                                # subsumed by matching_rules
                                target_cols=target_cols, target_descriptors=target_descriptors) # target_descriptor

                # --- TF-IDF model
                sv2, attr2, named_scores2 = \
                    fg_tfidf.compute_similarity_with_loinc(row, code, model=model, loinc_lookup=loinc_lookup, 
                        matching_rules=matching_rules, # this takes precedence over product(target_cols, target_descriptors)

                            # subsumed by matching_rules
                            target_cols=target_cols, target_descriptors=target_descriptors) # target_descriptors

                # --- Other models 

                if len(attributes) == 0: attributes = np.hstack([attr1, attr2])
                sv = np.hstack([sv1, sv2])
                pos_instances.append(sv)  # sv: a vector of similarity scores

                #########################################################################
                if verify: 
                    named_scores = named_scores2
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
                    if not tHasSignal: 
                        msg += "    + No similar properties found between {} and {} #\n".format(target_cols, target_descriptors)
                        print(msg)
                    if tHasSignal: 
                        highlight(show_evidence(row, sdict=named_scores, print_=False), symbol='#')
                #########################################################################

                if not tGenNegative: 
                    if n_codes == 1: print("[transform] Skipping negative examples ...")
                    continue

                # [Q] what happens if we were to assign an incorrect LOINC code, will T-attributes stay consistent with its LOINC descriptor? 
                codes_negative = loinc.sample_negatives(code, target_codes, n_samples=10, model=None, verbose=1)
                
                for code_neg in codes_negative: 

                    if code_neg in loinc_lookup: 

                        # --- String distance model
                        sv1, attr1, named_scores1 = \
                            fg_sdist.compute_similarity_with_loinc(row, code_neg, loinc_lookup=loinc_lookup, vars_lookup=vars_lookup,
                                matching_rules=matching_rules, # this takes precedence over product(target_cols, target_descriptors)
                                    add_sdist_vars=False,

                                        # subsumed by matching_rules
                                        target_cols=target_cols, target_descriptors=target_descriptors) # target_descriptor

                        # --- TF-IDF model
                        sv2, attr2, named_scores2 = \
                            fg_tfidf.compute_similarity_with_loinc(row, code_neg, model=model, loinc_lookup=loinc_lookup, 
                                matching_rules=matching_rules, # this takes precedence over product(target_cols, target_descriptors)

                                    # subsumed by matching_rules
                                    target_cols=target_cols, target_descriptors=target_descriptors) # target_descriptors
                        
                        sv = np.hstack([sv1, sv2])
                        neg_instances.append(sv)  # sv: a vector of similarity scores
                        
                        # ------------------------------------------------
                        if verify: 
                            named_scores = named_scores2

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
                                    highlight(show_evidence(row, code_neg=code_neg, sdict=named_scores, print_=False), symbol='#')
                        # ------------------------------------------------
    
    print("... There are n={} codes not found on the LONIC+MTRT corpus table:\n{}\n".format(len(codes_missed), codes_missed))

    if verify: 
        r_detected = n_detected/(n_comparisons_pos+0.0) if n_comparisons_pos > 0 else 0.0
        r_detected_in_neg = n_detected_in_negatives/(n_comparisons_neg+0.0) if n_comparisons_neg > 0 else 0.0
        print("...... Among N={} codes, r(detected): {}, r(detected in any -): {}".format(n_codes, r_detected, r_detected_in_neg))

    y = np.array([])
    if not tGenNegative: 
        # use this mode when transforming and predicting new examples
        X = np.array(pos_instances)
    else:
        X = np.vstack([pos_instances, neg_instances])
        y = np.hstack([np.repeat(1, len(pos_instances)), np.repeat(0, len(neg_instances))])

    print("[transform] from n(df)={}, we created n={} training instances".format(N0, X.shape[0]))
    
    if save: 
        pass  

    # note:        
    return X, y, attributes

def select_reliable_positive(cohort, method='classifier'):
    if method.startswith('class'): # use classifier array result as heuristics
        # assumption: LOINC codes with high performance scores are more reilable
        return select_loinc_codes_by_category(cohort=cohort, categories=['easy', ])
    elif method.startswith('sim'):  # similarity-based approach (e.g. similarity in LOINC strings)
        pass
    raise NotImplementedError

# select reliable positive via heuristics
def select_loinc_codes_by_category(cohort='hepatitis-c', categories=[]):
    from analyzer import label_by_performance

    if len(categories) == 0: 
        categories = ['easy', ]  # other categories: ['hard', 'low', ], where 'low': low sample size
    # use 'easy' codes to train
    # predict 'hard' and 'low'

    ccmap = label_by_performance(cohort=cohort, categories=categories)
    candidates = []
    for cat, codes in ccmap.items(): 
        # codes is in numpy array 
        candidates.extend(codes)

    non_codes = LoincTSet.null_codes # ['unknown', 'other', ]
    return list(set(candidates) - set(non_codes))

def demo_create_training_data(**kargs):
    def show_evidence(row, code_neg=None, sdict={}, print_=False, min_score=0.0, label='?'):
        # sdict: T-attribute -> Loinc descriptor -> score
         
        code = row[LoincTSet.col_code]
        msg = "(evidence) Found matching signals > code: {} ({})\n".format(code, label)
        if code_neg is not None:
            msg = "(evidence) {} ->? {} (-)\n".format(code, code_neg) 
            # ... the input code could be re-assigned to the code_neg

        for col, entry in sdict.items(): 
            msg += "... {}: {} ~ \n".format(col, row[col])  # a T-attribute and its value
            for col_loinc, score in entry.items():
                if score > min_score: 
                    msg += "    + {}: {} => score: {}\n".format(col_loinc, process_text(loinc_lookup[code][col_loinc]), score)
        if print_: print(msg)
        return msg

    from analyzer import label_by_performance, col_values_by_codes, load_src_data
    from feature_analyzer import plot_heatmap  

    cohort = kargs.get('cohort', 'hepatitis-c')

    # --- LOINC table attributes
    col_ln, col_sn = LoincTable.long_name, LoincTable.short_name
    col_lkey = LoincTable.col_key
    col_com = LoincTable.col_com
    col_sys = LoincTable.col_sys
    col_method = LoincTable.col_method
    col_prop = LoincTable.col_prop
    # -----------------------------------
    col_target = LoincTSet.col_target  # 'test_result_loinc_code'
    # -----------------------------------
    col_mval = LoincMTRT.col_value
    col_mkey = LoincMTRT.col_key       # loinc codes in the mtrt table

    # --- matching rules
    ######################################
    target_cols = ['test_order_name', 'test_result_name', ] # 'test_result_units_of_measure'
    # ... other fields: 'panel_order_name'

    target_descriptors = [col_sn, col_ln, col_com, col_sys, ]

    # note that sometimes we may also want to compare with MTRT
    matching_rules = { 'test_order_name': [col_sn, col_ln, col_com, col_sys, ], 
                       'test_result_name': [col_sn, col_ln, col_com, col_sys, col_prop, ], 
                       # 'test_specimen_type': [col_sys, ], 
                       # 'test_result_units_of_measure': [col_sn, col_prop], }
                       }
    ######################################

    # --- Cohort definition (based on target condition and classifier array performace)
    ######################################
    codesRP = select_reliable_positive(cohort, method='classifier')
    # RP: reliable positives 

    highlight("(demo) Create training data ... found n={} reliable positives".format(len(codesRP)))
    df_src = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=True, processed=False)
    codesAll = df_src[col_target].unique()
    ######################################

    # select just the reliable positive to generate matchmaker's training data
    # ... can deter this step until feature_transform
    ts = df_src[df_src[col_target].isin(codesRP)] # don't use this
    # ts = select_samples_by_loinc(ts, target_codes=codesRP, target_cols=target_cols)

    X_train, y_train, attributes = \
        feature_transform(ts, target_cols=target_cols, df_src=df_src, 
                matching_rules=matching_rules, 
                ngram_range=(1, 3), max_features=50000,  # TF-IDF model paramters

                target_codes=codesRP, 
                n_per_code=3,
                # ... generate feature vectors only on these LOINC codes (but redundant here since ts is already filtered accordingly)
                ) 

    # --- Visualize
    col_label = 'label'
    ts_train = DataFrame(X_train, columns=attributes)
    ts_train[col_label] = y_train

    n_display = 10
    vtype = subject = 'tfidf'

    # parentdir = os.path.dirname(os.getcwd())
    datadir = os.path.join(os.getcwd(), 'data')  # e.g. /Users/<user>/work/data
    output_file = f'matchmaker-train-{cohort}.csv'
    output_path = os.path.join(datadir, output_file)

    # Output
    # --------------------------------------------------------
    ts_train.to_csv(output_path, index=False, header=True)
    # --------------------------------------------------------

    tabulate(ts_train.sample(n=n_display), headers='keys', tablefmt='psql')

    # --- Create Test set
    highlight("(demo) Now create test data ... ")
    codesTest = select_loinc_codes_by_category(cohort='hepatitis-c', categories=['hard', 'low', ])
    
    # select target LOINC codes for the test set
    ts = df_src[df_src[col_target].isin(codesTest)]
    # ts = select_samples_by_loinc(ts, target_codes=codesTest target_cols=target_cols)
    # ... can deter this step until feature_transform

    X_test, _, _ = \
        feature_transform(ts, target_cols=target_cols, df_src=df_src, 
                matching_rules=matching_rules, 

                target_codes=codesTest, 
                n_per_code=3,

                gen_negative=False, # don't generate negative exmaples
                ngram_range=(1, 3), max_features=50000,  # TF-IDF model paramters

                ) 
    ts_test = DataFrame(X_test, columns=attributes)

    output_file = f'matchmaker-test-{cohort}.csv'
    output_path = os.path.join(datadir, output_file)

    # Output
    # --------------------------------------------------------
    ts_test.to_csv(output_path, index=False, header=True)
    # --------------------------------------------------------

    # train classifier

    return (ts_train, ts_test)

def test(**kargs):

    demo_create_training_data()

    return

if __name__ == "__main__": 
    test()