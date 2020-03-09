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
import language_model as langm
# from language_model import build_tfidf_model
import config

import common, text_processor
from text_processor import process_text, process_string
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

def partition(df, verbose=1, mode='prior'):
    """

    Memo
    ----
    1. "df" is the source data (not the transformed data), therefore we need to use 
            MatchmakerFeatureSet.categorize_features() 

            INSTEAD OF  

            MatchmakerFeatureSet.categorize_transformed_features()

    """
    # output: (dfM, dfX, dfY, dfD, dfZ)
    return MatchmakerFeatureSet.partition(df, verbose=verbose, mode=mode)

def tranform_and_encode(df, fill_missing=True, token_default='unknown', 
        drop_high_missing=False, pth_null=0.9, verbose=1):
    """
    Transform the non-matching variables (i.e. all variables NOT used to match
    with the LOINC decriptions such as meta_sender_name, test_order_code, test_result_code). 

    Matching variables are the T-attributes that carry text values (e.g. test_order_name, test_result_name)
    Non-matching variables are typically not text-valued columns

    Input
    -----
    df: source training data 

    """
    from analyzer import col_values
    from transformer import encode_vars 

    # matchmaker features 
    matching_cols = MatchmakerFeatureSet.matching_cols
    cat_cols = MatchmakerFeatureSet.cat_cols
    cont_cols = MatchmakerFeatureSet.cont_cols
    target_cols = MatchmakerFeatureSet.target_cols
    high_card_cols = MatchmakerFeatureSet.high_card_cols

    # --- transform variables
    FeatureSet.to_age(df)
    values = col_values(df, col='age', n=10)
    print("[transform] age: {}".format(values))
    # ... NOTE: raw feature transformation has to take place first, because it may introduce new variables

    # -- Categorize variables
    matching_vars, regular_vars, target_vars, derived_vars, meta_vars = \
        MatchmakerFeatureSet.categorize_features(df, remove_prefix=False)
    # ...note 
    #    regular_vars: non-matching columns (e.g. meta_sender_name, test_order_code, test_result_code, ...)

    # V = cont_cols + cat_cols  # + derived_cols (e.g. count)
    # L = target_cols
    dfM = df[matching_cols] # T-attributes that carry text values
    dfX = df[regular_vars]
    dfY = df[target_vars]

    # optinal variables
    dfD = df[derived_vars] if len(derived_vars) > 0 else DataFrame()
    dfZ = df[meta_vars] if len(meta_vars) > 0 else DataFrame()

    if fill_missing:  
        # dfM.fillna(value=token_default, inplace=True)
        # ... don't fill missing values for dfM here!
        dfX[cont_cols].fillna(value=0, inplace=True)
        dfX[cat_cols].fillna(value=token_default, inplace=True)
        # dfX.fillna(value=token_default, inplace=True)

    if drop_high_missing: 
        # drop columns/vars with too many missing values 
        N = dfX.shape[0]
        n_thresh = int(N * pth_null)
        nf0 = nf = dfX.shape[1]
        fset0 = set(dfX.columns.values)

        dfX = dfX[dfX.columns[dfX.isnull().mean() < pth_null]]
        fset = set(dfX.columns.values)
        nf = dfX.shape[1]
        print("[transform] Dropped n={} features:\n{}\n".format(nf-nf0, fset0-fset))

    dim0 = dfX.shape
    dfX, encoder = encode_vars(dfX, fset=cat_cols, high_card_cols=high_card_cols)
    if verbose: 
        print("[encode] dim(dfX-): {}, dim(dfX+): {}".format(dim0, dfX.shape))
        # [log] ... [encode] dim(dfX-): (64979, 13), dim(dfX+): (64979, 97)

    return (dfM, dfX, dfY, dfD, dfZ, encoder)

def regular_feature_transform(df, **kargs):
    tDropHighMissing = kargs.get('drop_high_missing', False)
    pth_null = kargs.get("pth_null", 0.9)  # threshold of null-value proportion to declare a "high missing rate"
    verbose = kargs.get('verbose', 1)
    token_default = kargs.get("token_default", LoincTSet.token_default)

    N0, Nv0 = df.shape
    dfM, dfX, dfY, dfD, dfZ, encoder = \
         tranform_and_encode(df, fill_missing=True, token_default=token_default, 
                drop_high_missing=tDropHighMissing, pth_null=pth_null)
    # ... transform and encode only deals with X i.e. regular variables (i.e. non-matching variables)
    
    df = pd.concat([dfM, dfX, dfY, dfD, dfZ], axis=1) # matching, regular, target, derived, meta  
    
    assert df.shape[0] == N0
    if verbose: highlight("[transform] dim of vars: {} -> {}".format(Nv0, df.shape[1]))
    return df

def text_feature_transform(df, **kargs): 
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
    def show_evidence(row, code, code_neg=None, sdict={}, print_=False, min_score=0.0, label='?'):
        # sdict: T-attribute -> Loinc descriptor -> score
         
        code = row[LoincTSet.col_code]
        code_x = code   # Target code's corresponding T-attributes are to be matched against 

        msg = "(evidence) Found matching signals for code(+): {} (target aka \"reliable\" positive)\n".format(code)
        if code_neg is not None:
            msg = "(hypothesis) Found matching signals when {}(+) -> {}(-)?\n".format(code, code_neg) 
            # ... the input code could be re-assigned to the code_neg
            label = '-'
            code_x = code_neg
        print("(debug) code_x: {}".format(code_x))
        for col, entry in sdict.items(): 
            msg += "... {}: {}\n".format(col, row[col])  # a T-attribute and its value
            for col_loinc, score in entry.items():  # how does the current row's T attributes compared to the LOINC code's descriptors?
                if score > min_score: 
                    msg += "... {}: {} => score: {} | {}({})\n".format(col_loinc, 
                        process_string(loinc_lookup[code_x][col_loinc]), score, code_x, label)
        if print_: print(msg)
        return msg
    def melt_rules(matching_rules): 
        t_attributes = set([])
        l_descriptors = set([])
        for tattr, descriptors in matching_rules.items(): 
            t_attributes.add(tattr)
            l_descriptors.update(descriptors)
        return list(t_attributes), list(l_descriptors)

    from analyzer import load_src_data
    from data_processor import toXY

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

    target_codes = kargs.get('target_codes', list(df[col_target].unique())) 

    loinc_lookup = kargs.get('loinc_lookup', {})
    vars_lookup = kargs.get('vars_lookup', {})
    verify = kargs.get('verify', True)
    save = kargs.get('save', False)

    # --- TF-IDF specific 
    ngram_range = kargs.get('ngram_range', (1,3))
    max_features = kargs.get('max_features', 50000)
    model = kargs.get("tfidf_model", None)

    # --- Data specific 
    tGenNegative = kargs.get('gen_negative', True)
    n_per_code = kargs.get("n_per_code", 3)
    tDropHighMissing = kargs.get('drop_high_missing', False)
    tAddRegularVars = kargs.get('add_regular_vars', False)

    # --- matching process 
    tProcessText = kargs.get('process_text', True)
    tRemoveDupTokens = kargs.get('remove_dup', True)
    matching_rules = kargs.get('matching_rules', {})

    dfM, dfX, dfY, df_derived, df_meta = partition(df)  # df: is source data
    df = pd.concat([dfM, dfY], axis=1)
    # ... df: redefined to only include matchmaking variables + target (e.g. LOINC code)
    # ... why? because then we could append dfX as a separate process to incorporate non-matching variables when desired

    # --- Define matching rules --- 
    ######################################
    target_cols = kargs.get('target_cols', ['test_order_name', 'test_result_name', ]) # options: 'test_result_units_of_measure',

    print("[transform] The following T-attributes are to be compared with LOINC descriptors:\n{}\n".format(target_cols))
    assert np.all(col in df.columns for col in target_cols)

    # ... other fields: 'panel_order_name'
    target_descriptors = kargs.get('target_loinc_cols', [col_sn, col_ln, col_com, col_sys, ])
    if not matching_rules: matching_rules = {target_col: target_descriptors for target_col in target_cols}    
    # e.g.
    # matching_rules = {'test_order_name': [col_sn, col_ln, col_com, col_sys, ], 
    #                   'test_result_name': [col_sn, col_ln, col_com, col_sys, col_prop, ], 
    #                   'test_specimen_type': [col_sys, ], 
    #                   'test_result_units_of_measure': [col_sn, col_prop], 
    #                   }
    ######################################

    # default to use all LOINC codes associatied with the given cohort as the source of training corpus
    if model is None: 
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
       vars_lookup = LoincTSet.load_sdist_var_descriptors(target_cols, process_text=tProcessText, remove_dup=tRemoveDupTokens)
       # ... by default, process_string is invoked on the T-attributes

       assert len(vars_lookup) == len(target_cols)
       print("[transform] String-distance variables | size(vars_lookup): {}".format(len(vars_lookup)))

    codes_missed = set([])
    n_codes = 0
    n_comparisons_pos = n_comparisons_neg = 0 
    n_detected = n_detected_in_negatives = 0
    
    # training data based on the matching between T-attributes and LOINC descriptors
    ##################################
    pos_instances = []
    neg_instances = []

    # meta data
    ##################################

    # init meta data schema
    pos_label = MatchmakerFeatureSet.pos_label  # 1
    neg_label = MatchmakerFeatureSet.neg_label  # 0
    stypes = [pos_label, neg_label]  # sample types
    col_assignment = 'assignment'
    meta_data = {stype:{} for stype in stypes}
    target_cols, target_descriptors = melt_rules(matching_rules)
    meta_attributes = MatchmakerFeatureSet.meta_cols 
    # [col_assignment, ] + target_cols + target_descriptors  # test_result_loinc_code, test_order_name, test_result_name, ... 
    
    meta_attributes_tracked = MatchmakerFeatureSet.customize_meta_cols(target_cols, target_descriptors)
    for stype, entry in meta_data.items(): 
        meta_data[stype] = {ma: [] for ma in meta_attributes_tracked}
  
    # LOINC 
    # pos_codes = []
    # neg_codes = []

    df_index = []   # data index
    ##################################

    attributes = []
    N0 = df.shape[0]
    feature_suffices = MatchmakerFeatureSet.models # ['sdist', 'tfidf', ] 
    for code, dfc in df.groupby([LoincTSet.col_code, ]):
        n_codes += 1

        if n_codes % 10 == 0: print("[transform] Processing code #{}: {}  ...".format(n_codes, code))
        if code in LoincTSet.null_codes: continue
        if not code in loinc_lookup: 
            codes_missed.add(code)  

        # indices = dfc.index.values
        for r, row in dfc.iterrows():   # r ~ df.index
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

                # a. regular variables
                row_X = dfX.iloc[r]   # df now does not have the same index as dfX
                sv3 = row_X.values
                attr3 = dfX.columns.values

                if len(attributes) == 0: 
                    # attributes = np.hstack([attr1, attr2])
                    attributes = FeatureSet.join_features([attr1, attr2], feature_suffices)
                    if tAddRegularVars: 
                        print("[transform] (before) feature set (n={}):\n{}\n".format(len(attributes), attributes))
                        attributes = np.hstack([attr3, attributes])
                        print("[transform] (after)  feature set (n={}):\n{}\n".format(len(attributes), attributes))

                sv = np.hstack([sv1, sv2, sv3]) if tAddRegularVars else np.hstack([sv1, sv2])
                pos_instances.append(sv)  # sv: a vector of similarity scores

                # keep track of meta data (e.g. LOINC assignment itself)
                meta_data[pos_label][col_assignment].append(code)
                for tcol in MatchmakerFeatureSet.matching_cols:
                    if tcol in target_cols: 
                        meta_data[pos_label][tcol].append(row[tcol]) 
                for dcol in MatchmakerFeatureSet.descriptors: 
                    if dcol in target_descriptors: 
                        meta_data[pos_label][dcol].append(loinc_lookup[code][dcol])
                # df_index.append(r)  # ... not useful for keeping track of negative samples

                #########################################################################
                if verify: 
                    named_scores = named_scores1
                    # positive_scores = defaultdict(dict)  # collection of positive sim scores, representing signals
                    tHasSignal = False
                    msg = f"[{r}] Code(+): {code}\n"
                    for target_col, entry in named_scores.items(): 
                        msg_t =  "... Col: {}: {}\n".format(target_col, process_string(row[target_col]))
                        msg_t += "... SN:  {}: {}\n".format(code, process_string(loinc_lookup[code][col_sn]))
                        msg_t += "... LN:  {}: {}\n".format(code, process_string(loinc_lookup[code][col_ln]))
                        
                        for target_dpt, score in entry.items():
                            n_comparisons_pos += 1
                            if score > 0: 
                                n_detected += 1
                                msg += msg_t
                                msg += "    + {}: {}\n".format(target_dpt, score)
                                # nonzeros.append((target_col, target_dpt, score))
                                # positive_scores[target_col][target_dpt] = score
                                tHasSignal = True
                    # ------------------------------------------------
                    if not tHasSignal: 
                        msg += "... Code(+): {} | No similar properties found between row attributes: {} and its LOINC dpt: {} #\n".format(code, target_cols, target_descriptors)
                        print(msg); msg = ""
                    if tHasSignal: 
                        highlight(show_evidence(row, code=code, sdict=named_scores, print_=False), symbol='#')
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
 
                        # --- Other models 

                        # a. regular variables
                        row_X = dfX.iloc[r]   # df now does not have the same index as dfX
                        sv3 = row_X.values
                        attr3 = dfX.columns.values
                        # ... original row attributes remain fixed regardless of the negative LOINC code
                        
                        sv = np.hstack([sv1, sv2, sv3]) if tAddRegularVars else np.hstack([sv1, sv2])
                        neg_instances.append(sv)  # sv: a vector of similarity scores

                        # keep track of meta data
                        meta_data[neg_label][col_assignment].append(code_neg)
                        for tcol in MatchmakerFeatureSet.matching_cols:
                            if tcol in target_cols: 
                                meta_data[neg_label][tcol].append(row[tcol])  # this is the same as the postives ... 
                        for dcol in MatchmakerFeatureSet.descriptors: 
                            if dcol in target_descriptors: 
                                meta_data[neg_label][dcol].append(loinc_lookup[code_neg][dcol]) # ... but now the assignment changes to the negative
                        
                        # ------------------------------------------------
                        if verify: 
                            named_scores = named_scores1

                            tHasSignal = False
                            # positive_scores = defaultdict(dict)
                            msg = title = f"[{r}] Code(-): {code_neg} ... if we deliberately assign this code, what happens?\n"
                            for target_col, entry in named_scores.items(): 
                                msg_t =  "... Col: {}: {}\n".format(target_col, process_string(row[target_col]))
                                msg_t += "... SN:  {}: {}\n".format(code_neg, process_string(loinc_lookup[code_neg][col_sn]))
                                msg_t += "... LN:  {}: {}\n".format(code_neg, process_string(loinc_lookup[code_neg][col_ln]))
                                

                                # nonzeros = []
                                for target_dpt, score in entry.items():
                                    n_comparisons_neg += 1
                                    if score > 0: 
                                        n_detected_in_negatives += 1
                                        msg += msg_t
                                        msg += "    + {}: {}\n".format(target_dpt, score)
                                        # positive_scores[target_col][target_dpt] = score
                                        tHasSignal = True

                                if tHasSignal: 
                                    msg += "... Code(-) {} | Found similar properties between T-attributes(code={}) and negative LOINC dpt: {}  ###\n".format(code_neg, code, code_neg)
                                    print(msg); msg = ""
                                if tHasSignal: 
                                    highlight(show_evidence(row, code=code, code_neg=code_neg, sdict=named_scores, print_=False), symbol='#')
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
        y = np.hstack([np.repeat(pos_label, len(pos_instances)), np.repeat(neg_label, len(neg_instances))])

    print("[transform] from n(df)={}, we created n={} training instances.".format(N0, X.shape[0]))

    # meta_data['assignment'] = np.hstack(pos_codes, neg_codes)
    # assert len(meta_data['assignment']) == len(y)
    
    if save: 
        pass  

    # note:        
    return X, y, attributes, meta_data

def text_feature_transform2(df, **kargs): 
    def add_meta_data(ts, meta_data={}, is_test_data=False, col_target='', token_default=""):
        if not col_target: col_target = MatchmakerFeatureSet.col_target
        pos_label = MatchmakerFeatureSet.pos_label
        neg_label = MatchmakerFeatureSet.neg_label

        labels = [pos_label, neg_label]
        assert np.all([label in meta_data for label in labels]), "labels: {}, meta_keys: {}".format(list(labels), list(meta_data.keys()))

        if is_test_data: 
            
            ts_meta_pos = DataFrame(meta_data[pos_label], columns=meta_data[pos_label].keys())
            # ts_meta_neg = DataFrame({}, columns=meta_data[pos_label].keys()) 

            ts_meta = ts_meta_pos

        else:

            ts_pos = ts[ts[col_target]==pos_label]
            ts_neg = ts[ts[col_target]==neg_label]

            ts_meta_pos = DataFrame(meta_data[pos_label], columns=meta_data[pos_label].keys())
            ts_meta_neg = DataFrame(meta_data[neg_label], columns=meta_data[neg_label].keys())
            assert ts_meta_pos.shape[1] == ts_meta_neg.shape[1]

            # we do not necessarily have negative examples (e.g. test data
            try: 
                nRef =  next(iter(meta_data[1].values())) # len(meta_data[1]['test_order_name'])
            except: 
                nRef = -1
                print("[transform2] Could not iterate meta data:\n{}\n".format(meta_data))
            assert ts_meta_pos.shape[0] == ts_pos.shape[0], "ts_meta_pos(n={}) <> ts_pos(n={}) | n(ref): {}".format(
                         ts_meta_pos.shape[0], ts_pos.shape[0], nRef)
            assert ts_meta_neg.shape[0] == ts_neg.shape[0], "ts_meta_neg(n={}) <> ts_neg(n={})".format(ts_meta_neg.shape[0], ts_neg.shape[0])
     
            ts_meta = pd.concat([ts_meta_pos, ts_meta_neg], ignore_index=True)

        # ... now we have the meta dataframe ready

        # fill missing
        ts_meta.fillna(value=token_default, inplace=True)
    
        return pd.concat([ts, ts_meta], axis=1)

    col_label = kargs.get('label', MatchmakerFeatureSet.col_target)
    tAddMetaData = kargs.get("add_meta_data", True)
    label_unknown = kargs.get("label_placeholder", -1) 

    X, y, attributes, meta_data = text_feature_transform(df, **kargs)
    ts = DataFrame(X, columns=attributes)

    isTestData = False
    if len(y) > 0: 
        assert len(y) == X.shape[0]
        ts[col_label] = y
    else: 
        # test data
        ts[col_label] = label_unknown
        isTestData = True

    if tAddMetaData: 
        ts = add_meta_data(ts, meta_data, is_test_data=isTestData)
    
    return ts

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
    def show_evidence(row, code, code_neg=None, sdict={}, print_=False, min_score=0.0, label='?'):
        # sdict: T-attribute -> Loinc descriptor -> score
         
        code = row[LoincTSet.col_code]
        code_x = code   # Target code's corresponding T-attributes are to be matched against 

        msg = "(evidence) Found matching signals for code(+): {} (target aka \"reliable\" positive)\n".format(code)
        if code_neg is not None:
            msg = "(hypothesis) Found matching signals when {}(+) -> {}(-)?\n".format(code, code_neg) 
            # ... the input code could be re-assigned to the code_neg
            label = '-'
            code_x = code_neg

        for col, entry in sdict.items(): 
            msg += "... {}: {}\n".format(col, row[col])  # a T-attribute and its value
            tSignal = False
            for col_loinc, score in entry.items():  # how does the current row's T attributes compared to the LOINC code's descriptors?
                if score > min_score: 
                    msg += "... {}: {} => score: {} | {}({})\n".format(col_loinc, 
                        process_string(loinc_lookup[code_x][col_loinc]), score, code_x, label)
                    tSignal = True 
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

    # matchmaker training data parameters 
    col_pos_assign = "pos_code"   # positive assignment (LOINC being correct)
    col_neg_assign = "neg_code"   # negative assignment

    # Data parameters 
    tProcessedTAttributes = False
    tAddRegularVars = False
    tDropHighMissing = True
    tAddMetaData = True
    tShuffle = True
    token_default = ""

    # TF-IDF parameters
    ngram_range = kargs.get('ngram_range', (1,3))
    max_features = kargs.get('max_features', 50000)

    # --- matching rules
    ######################################
    target_cols = ['test_order_name', 'test_result_name', ] # 'test_result_units_of_measure'
    # ... other fields: 'panel_order_name'

    target_descriptors = [col_sn, col_ln, col_com, col_sys, ]

    # note that sometimes we may also want to compare with MTRT
    matching_rules = kargs.get('matching_rules', 
                            { 'test_order_name': [col_sn, col_ln, col_com, ],  # col_sys
                               # 'test_result_name': [col_sn, col_ln, col_com,  ], #  col_sys, col_prop
                               # 'test_specimen_type': [col_sys, ], 
                               # 'test_result_units_of_measure': [col_sn, col_prop], }
                               }
                       )
    ######################################

    # --- Cohort definition (based on target condition and classifier array performace)
    ######################################
    codesRP = select_reliable_positive(cohort, method='classifier')
    # RP: reliable positives 

    highlight("(demo) Create training data ... found n={} reliable positives".format(len(codesRP)))
    df_src = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=True, processed=tProcessedTAttributes)
    df_src = regular_feature_transform(df_src, drop_high_missing=False, pth_null=0.9)
    # ... drop_high_missing only work on non-matchmaking variables
    # ... todo: load_transform
    print("(demo) After reg feature transform | cols(df):\n{}\n".format(list(df_src.columns)))

    codesAll = df_src[col_target].unique()
    ######################################

    # traing TF-IDF model
    tfidf_model = build_tfidf_model(cohort=cohort, df_src=df_src, target_cols=target_cols, 
                        ngram_range=ngram_range, max_features=max_features)

    # select just the reliable positive to generate matchmaker's training data
    # ... can deter this step until feature_transform
    ts = df_src.loc[df_src[col_target].isin(codesRP)] # don't use this
    # ts = select_samples_by_loinc(ts, target_codes=codesRP, target_cols=target_cols)
   
    # X_train, y_train, attributes, meta_data = \
    ts_train = text_feature_transform2(ts, target_cols=target_cols, df_src=df_src, 
                        matching_rules=matching_rules, 

                        tfidf_model=tfidf_model, 
                        ngram_range=ngram_range, max_features=max_features,  # TF-IDF model paramters

                        target_codes=codesRP, 
                        # ... generate feature vectors only on these LOINC codes (but redundant here since ts is already filtered accordingly)
                        n_per_code=3,

                        add_regular_vars=tAddRegularVars,

                        # wrapper parameters 
                        add_meta_data=tAddMetaData

                    ) 

    if tShuffle: 
        ts_train = ts_train.sample(frac=1).reset_index(drop=True)

    highlight("(demo) Created n={} training instances with m={} attributes:\n{}\n".format(ts_train.shape[0], 
        ts_train.shape[1], ts_train.columns.values))
    
    n_display = 10
    vtype = subject = 'tfidf'

    # parentdir = os.path.dirname(os.getcwd())
    datadir = os.path.join(os.getcwd(), 'data')  # e.g. /Users/<user>/work/data
    output_file = f'matchmaker-train-{cohort}.csv'
    output_path = os.path.join(datadir, output_file)

    # Output
    # --------------------------------------------------------
    print("(demo) Saving training data to:\n{}\n".format(output_path))
    ts_train.to_csv(output_path, index=False, header=True)
    # --------------------------------------------------------

    tabulate(ts_train.sample(n=n_display), headers='keys', tablefmt='psql')

    # --- Create Test set
    highlight("(demo) Now create test data ... ")
    codesTest = select_loinc_codes_by_category(cohort='hepatitis-c', categories=['hard', 'low', ])
    
    # select target LOINC codes for the test set
    ts = df_src[df_src[col_target].isin(codesTest)]
    # ts = select_samples_by_loinc(ts, target_codes=codesTest target_cols=target_cols)
    # ... can deter this step until text_eature_transform

    # X_test, y_test, _, meta_data = \
    ts_test = text_feature_transform2(ts, target_cols=target_cols, df_src=df_src, 
                    matching_rules=matching_rules, 

                    tfidf_model=tfidf_model,
                    ngram_range=ngram_range, max_features=max_features,  # TF-IDF model paramters

                    target_codes=codesTest, 
                    n_per_code=3,

                    gen_negative=False, # don't generate negative exmaples
                    add_regular_vars=tAddRegularVars, 

                    # wrapper parameters 
                    add_meta_data=tAddMetaData

                ) 
    print("(demo) dim(ts_test): {}".format(ts_test.shape))

    output_file = f'matchmaker-test-{cohort}.csv'
    output_path = os.path.join(datadir, output_file)

    # Output
    # --------------------------------------------------------
    print("(demo) Saving test data to:\n{}\n".format(output_path))
    ts_test.to_csv(output_path, index=False, header=True)
    # --------------------------------------------------------

    # train classifier

    return (ts_train, ts_test)

def visualize_training_data(ts, **kargs): 
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
    
    tScale = False # set zcore = 1 within clustermap() instead
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
    col_label = kargs.get('col_label', MatchmakerFeatureSet.col_target)
    cohort = kargs.get('cohort', config.cohort)
    vtype = subject = kargs.get('vtype', 'combined')
    n_samples = kargs.get('n_samples', 50)

    cols_y = [col_label, ]
    cols_untracked = []
    # ---------------------------------------------

    # -- Filter unwanted variables such as meta data
    matching_vars, regular_vars, target_vars, derived_vars, meta_vars = \
        MatchmakerFeatureSet.categorize_transformed_features(ts, remove_prefix=False)

    # -- here, we are only interested in matching variables
    highlight("(demo) Found n={} matching vars:\n{}\n".format(len(matching_vars), matching_vars))
    ts = ts.drop(meta_vars, axis=1) # [matching_vars+target_vars]

    # n_samples: limit sample size to unclutter plot 
    df_match = down_sample(ts, col_label=col_label, n_samples=n_samples)
    df_match = df_match.sort_values(by=[col_label, ], ascending=True)
    df_match = relabel(df_match)
    # ---------------------------------------------

    df_pos = df_match[df_match[col_label]==1]
    df_neg = df_match[df_match[col_label]==0]
    highlight("(demo) dim(+): {}, dim(-): {}".format(df_pos.shape, df_neg.shape))

    # -- perturb the negative examples by a small random noise
    # ep = np.min(df_pos.values[df_pos.values > 0])
    # df_neg += np.random.uniform(ep/100, ep/10, df_neg.shape)
    # print("... ep: {} => perturbed df_neg:\n{}\n".format(ep, df_neg.head(10)))
    # df_match = pd.concat([df_pos, df_neg])

    # ---------------------------------------------
    # labels = df_match.pop('match_status')  # this is an 'inplace' operation
    labels = df_match[col_label]  # labels: a Series
    n_labels = np.unique(labels.values)
    # ---------------------------------------------

    lut = {0: "#3933FF", 1: "#FF3368"} # dict(zip(labels.unique(), "rb"))
    # positive (red): #3933FF, #3358FF, #e74c3c
    # negative (blue): #FF3368,  #3498db 
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

    X, y, fset, lset = toXY(df_match, cols_y=cols_y, scaler=None, perturb=False)
    if tScale:
        print("... feature set: {}".format(fset))

        X = common.scale(X, scaler='minmax') # standardize ~ z-score, normalize: minmax
        # df_match = DataFrame(np.hstack([X, y, z]), columns=fset+lset+cols_untracked)
        dfX = DataFrame(X, columns=fset)
        # ... don't include y here

        df_zeros = dfX.loc[(dfX.T == 0).all()]
        print("... After standardization, found n={} zero vectors".format(df_zeros.shape[0]))
    else: 
        dfX = df_match.drop(cols_y, axis=1)

    if tPerturb: 
        X = common.perturb(X, lower_bound=0, alpha=10.)
    # ... (dfX, X, y, fset, lset)

    #-----------------------------------------
    # ... test
    print("... dfX:\n{}\n".format(dfX))
    # X = common.scale(X, scaler='normalize') # standardize ~ z-score, normalize: minmax
    y = y.flatten()
    print("(test) min(X): {}, max(X): {}".format(np.min(X), np.max(X)))
    print("... dim(X): {}, dim(y): {}".format(X.shape, y.shape))
    print("... column-wise mean (+):\n{}\n".format(list(zip(fset, np.mean(X[y==1, :], axis=0)))))
    print("... column-wise mean (-):\n{}\n".format(list(zip(fset, np.mean(X[y==0, :], axis=0)))))
    print("... column-wise mean (A):\n{}\n".format(list(zip(fset, np.mean(X, axis=0)))))
    row_mean = np.mean(X, axis=1)
    print("... n(row-wise 0): {}".format(np.sum(row_mean == 0)))
    #-----------------------------------------

    highlight("(demo) 1. Plotting dendrogram (on top of heatmap) | dim(dfX): {}...".format(dfX.shape), symbol='#')

    # Normalize the data within the rows via z_score
    sys.setrecursionlimit(10000)
    try: 
        g = sns.clustermap(dfX, figsize=(15, 24.27), z_score=1, row_colors=row_colors, 
                   cmap='coolwarm', metric='cosine', method='complete')  # z_score=1, standard_scale=1
    except Exception as e:
        df_match = filter_features(df_match)
        dfX = df_match.drop(cols_y, axis=1)
        g = sns.clustermap(dfX, figsize=(15, 24.27), row_colors=row_colors, 
                   cmap='coolwarm', metric='cosine', method='complete')  # z_score=1, standard_scale=1

    # ... need to pass dataframe dfX instead of X! 
    # ... try different cmaps: 'vlag'

    # g.set_xticklabels(g.get_xticklabels(), rotation=45)
    # print(dir(g.ax_heatmap))
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=75)
    # plt.setp(g.ax_heatmap.secondary_yaxis(), rotation=45)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=16, fontweight='light')
    # ... horizontalalignment='right' shifts labels to the left
    
    # colors:  cmap="vlag", cmap="mako", cmap=palette
    # normalization: z_score, stndardize_scale

    output_path = os.path.join(config.plot_dir, f'clustermap-{vtype}-{cohort}.pdf')
    # g.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight') 
    saveFig(plt, output_path, dpi=300)

    ################################################
    # df_match = pd.read_csv(input_path, sep=",", header=0, index_col=None, error_bad_lines=False)
    df_match = down_sample(ts, col_label=col_label, n_samples=n_samples)
    df_match = relabel(df_match)
    df_pos = df_match[df_match[col_label]==1]
    df_neg = df_match[df_match[col_label]==0]
    print("... cols(df_pos):\n{}\n".format(df_pos.columns.values))
    df_pos.drop([col_label, ], axis=1, inplace=True)
    df_neg.drop([col_label, ], axis=1, inplace=True)
    # ... no X scaling

    # --- Enhanced heatmap 
    highlight("(demo) 2. Visualize feature values > (+) examples should have higher feature values, while (-) have low to zero values")
    output_path = os.path.join(config.plot_dir, f'{vtype}-pos-match-{cohort}.png')
    plot_data_matrix(df_pos, output_path=output_path, dpi=300)

    output_path = os.path.join(config.plot_dir, f'{vtype}-neg-match-{cohort}.png')
    plot_data_matrix(df_neg, output_path=output_path, dpi=300)

    #################################################
    # --- PCA plot 

    plt.clf()
    highlight("(demo) 3. PCA plot", symbol='#')
    
    # X = df_match.drop(cols_y+untracked, axis=1).values

    # read in the data again 
    # df_match = pd.read_csv(input_path, sep=",", header=0, index_col=None, error_bad_lines=False)
    df_match = down_sample(ts, col_label=col_label)
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

    output_path = os.path.join(config.plot_dir, f'PCA-{vtype}-{cohort}.png') 
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

    output_path = os.path.join(config.plot_dir, f'tSNE-{vtype}-{cohort}.png') 
    saveFig(plt, output_path, dpi=300)

    return

def save_dataset(ts, dtype='test', **kargs): 
    import data_processor as dp

    cohort = kargs.get('cohort', config.cohort)
    output_dir = kargs.get("data", config.data_dir)  # e.g. /Users/<user>/work/data
    output_file = kargs.get("output_file", "")
    sep = kargs.get('sep', ",")
    suffix = kargs.get("suffix", "")
    verbose = kargs.get("verbose", 1)

    if not output_file: 
        output_file = "matchmaker-{}-{}-{}.csv".format(dtype, cohort, suffix) if len(str(suffix)) > 0 else "matchmaker-{}-{}.csv".format(dtype, cohort)
    print("[save] Saving matchmaker data (dtype={}, suffix={}) as:\n{}\n".format(dtype, suffix, output_file))

    dp.save_generic(ts, output_file=output_file, output_dir=output_dir, 
        # parameters below are only for display
        dtype=dtype, sep=sep, verbose=verbose) 

    # if verbose: print("[save] Saving (transformed) data (dtype={}, dim={}) to:\n{}\n".format(dtype, ts.shape, output_path))
    # ts.to_csv(output_path, sep=sep, index=False, header=True) 
    return

# [todo] refactor to Matchmaker class
def load_dataset(dtype, **kargs): 
    cohort = kargs.get('cohort', config.cohort)
    input_dir = kargs.get("data", config.data_dir)  # e.g. /Users/<user>/work/data
    sep = kargs.get('sep', ",")
    verbose = kargs.get("verbose", 1)

    tMatchingVarsOnly = kargs.get("matching_vars_only", False) # if False, will also include regular variables if they exist
    tIncludeMeta = kargs.get("include_meta", False)
    tIncludeDerived = kargs.get("include_derived", False)
    tDropUntracked = kargs.get("drop_untracked", True)   # untracked, to be updated everytime when a new model is trained
    if tIncludeMeta or tIncludeDerived: tMatchingVarsOnly = False

    matching_cols = kargs.get("matching_cols", ['test_order_name', 'test_result_name'])

    input_path = os.path.join(input_dir, "matchmaker-{}-{}.csv".format(dtype, cohort))  

    ts = pd.read_csv(input_path, sep=sep, header=0, index_col=None, error_bad_lines=False)
 
    if tDropUntracked: ts = MatchmakerFeatureSet.drop_untracked(ts, verbose=1)

    matching_vars, regular_vars, target_vars, derived_vars, meta_vars = \
        MatchmakerFeatureSet.categorize_transformed_features(ts, remove_prefix=False)
    # assert len(matching_vars) > len(matching_cols)
    print("[load] matching_vars: {}".format(matching_vars))
    print("...    regular_vars:  {}".format(regular_vars))
    print("...    target vars:   {}".format(target_vars))
    print("...    meta_vars:     {}".format(meta_vars))

    if tMatchingVarsOnly: 
        tsX = ts[matching_vars]
        tsY = ts[target_vars]
        ts = pd.concat([tsX, tsY], axis=1)

    else:
        tsX = ts[matching_vars+regular_vars]
        tsY = ts[target_vars]

        # meta variables and derived variables
        tsM = tsD = DataFrame()
        if tIncludeMeta and len(meta_vars) > 0: 
            if verbose: print("(load_dataset) Found meta data: {}".format(meta_vars))
            tsM = ts[meta_vars]
        if tIncludeDerived and len(derived_vars) > 0: 
            if verbose: print("(load_dataset) Found derived variables: {}".format(derived_vars))
            tsD = ts[derived_vars]

        ts = pd.concat([tsX, tsY, tsD, tsM], axis=1)

    print("[load] Load (transformed) data (dtype={}, dim={}) from:\n{}\n".format(dtype, ts.shape, input_path))
    return ts

def filter_features(ts, scaler=None):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity

    # matching_cols = kargs.get("matching_cols", ['test_order_name', 'test_result_name'])
    print("[filter] cols(ts):\n{}\n".format(ts.columns.values))
    matching_vars, regular_vars, target_vars, derived_cols, meta_cols = MatchmakerFeatureSet.categorize_transfromed_features(ts)
    fX = matching_vars + regular_vars
    print("... matching_vars:\n{}\n".format(matching_vars))

    Nv0 = len(fX)
    N0 = ts.shape[0]
    # tsX = ts[[col for col in fX if np.mean(ts[col].values) > 0]]
 
    mms = MinMaxScaler()
    std_scale = StandardScaler()
    # for col in fX: 
    #     vmin, vmax = np.min(ts[col].values), np.max(ts[col].values)
    ts[fX] = std_scale.fit_transform(ts[fX])
    tsX = ts[[col for col in fX if not ts[col].isnull().values.any()]]
    
    tsY = ts[target_vars]
    ts = pd.concat([tsX, tsY], axis=1)
    print("[filter] After filtering 0 vars | n(vars): {} -> {}".format(Nv0, tsX.shape[1]))

    # filter 0 rows 
    row_mean = np.mean(tsX.values, axis=1)
    ts = ts.loc[row_mean != 0]
    # ts = ts.loc[~(ts.T == 0).all()]
    print("[filter] After filtering 0 rows | n(ts): {} -> {}".format(N0, ts.shape[0]))

    # pairwise similarity, computable? 
    A = cosine_similarity(tsX.values)
    print("[filter] Computed pairwise cosine similarity ...")

    return ts

def demo_fearture_analysis(**kargs):
    from analyzer import balance_by_downsampling


    cohort = kargs.get('cohort', "hepatitis-c")
    tAddRegularVars = False

    # load training data
    ts_train = load_dataset(dtype='train', include_derived=False, include_meta=False) 
    # ... visualization does not require meta variables
    
    # ts_train = filter_features(ts_train) 

    # ts_train = balance_by_downsampling2(ts_train, cols_y=[LoincTSet.col_target, ], method='multiple', majority_max=1, verify=1)
    visualize_training_data(ts_train, n_samples=50, verbose=1)
    # fg_tfidf.demo_create_vars_part2(df_match=ts_train)

    return 

def test(**kargs):

    ### Generate training data
    demo_create_training_data()

    ### Analyze features
    demo_fearture_analysis()

    return

if __name__ == "__main__": 
    test()