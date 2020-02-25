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
from language_model import build_tfidf_model
import config

import common, text_processor
from text_processor import process_text
from CleanTextData import standardize 

import feature_gen_tfidf as fg_tfidf
import feature_gen_sdist as fg_sdist

def feature_transform(df, target_cols=[], df_src=None, **kargs): 
    """
    Convert T-attributes into TF-IDF feature matrix with respect to the LOINC descriptors

    df -> X

    Params
    ------ 
    df: the data set containing the positive examples (with reliable LOINC assignments)

    """
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

    cohort = kargs.get('cohort', 'hepatitis-c')  # determines training data set
    target_codes = kargs.get('target_codes', []) 
    loinc_lookup = kargs.get('loinc_lookup', {})
    vars_lookup = kargs.get('vars_lookup', {})
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
                      'test_result_units_of_measure': [col_sn, col_method]
                      }
    ######################################
    highlight("Gathering training corpus (by default, use all data assoc. with target cohort: {} ...".format(cohort), symbol='#')
    if df_src is None: df_src = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=True, processed=True)
    col_new = 'corpus'
    # df_corpus = load_corpus(domain=cohort)
    df_corpus = get_corpora_by_loinc(df_src, target_cols, add_loinc_mtrt=True, 
                processed=True, dehyphenate=True, verbose=1, return_dataframe=True, col_new=col_new, save=False)
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

    if not vars_lookup: 
       vars_lookup = LoincTSet.load_sdist_var_descriptors(target_cols)
       assert len(vars_lookup) == len(target_cols)
       print("[transform] size(vars_lookup): {}".format(len(vars_lookup)))

    # gen_sim_features(df_src=df_src, df_loinc=None, df_map=None, transformed_vars_only=True, verbose=1) 
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
                    if not tHasSignal: msg += "    + No similar properties found between {} and {} #\n".format(target_cols, target_descriptors)
                    print(msg)
                    if tHasSignal: 
                        highlight(show_evidence(row, sdict=named_scores, print_=False), symbol='#')
                #########################################################################

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
                                    highlight(show_evidence(row, code_neg=code_neg, sdict=positive_scores, print_=False), symbol='#')
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
            processed=True, dehyphenate=True, verbose=1, return_dataframe=True, col_new=col_new, save=False)
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
                if len(positive_scores) == 0: msg += "    + No similar properties found between {} and {} #\n".format(target_cols, target_descriptors)
                print(msg)
                if len(positive_scores) > 0: 
                    highlight(show_evidence(row, sdict=positive_scores, print_=False), symbol='#')

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
                                highlight(show_evidence(row, code_neg=code_neg, sdict=positive_scores, print_=False), symbol='#')
                if tFoundMatchInNeg: 
                    n_detected_in_negatives += 1

    print("... There are n={} codes not found on the LONIC+MTRT corpus table:\n{}\n".format(len(codes_missed), codes_missed))
    r_detected = n_detected/(n_comparisons_pos+0.0)
    r_detected_in_neg = n_detected_in_negatives/(n_comparisons_neg+0.0)
    print("...... Among N={} codes, r(detected): {}, r(detected in any -): {}".format(n_codes, r_detected, r_detected_in_neg))
    
    # --- Visualize
    df_pos = DataFrame(pos_instances, columns=attributes)
    df_pos['label'] = 1
    df_neg = DataFrame(neg_instances, columns=attributes)
    df_neg['label'] = 0
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
    
    cohort = kargs.get('cohort', 'hepatitis-c')
    tStandardize = False

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

    cols_y = ['label', ]
    cols_untracked = []
    # ---------------------------------------------

    # read the feature vectors
    parentdir = os.path.dirname(os.getcwd())
    testdir = os.path.join(parentdir, 'test')  # e.g. /Users/<user>/work/data
    input_file = f'{vtype}-vars.csv'
    input_path = os.path.join(testdir, input_file)
    df_match = pd.read_csv(input_path, sep=",", header=0, index_col=None, error_bad_lines=False)
    # ---------------------------------------------

    # limit sample size to unclutter plot 
    n_samples = 50
    df_match = down_sample(df_match, col_label='label', n_samples=n_samples)
    df_match = df_match.sort_values(by=['label', ], ascending=True)
    df_match = relabel(df_match)
    # ---------------------------------------------

    df_pos = df_match[df_match['label']==1]
    df_neg = df_match[df_match['label']==0]

    # -- perturb the negative examples by a small random noise
    # ep = np.min(df_pos.values[df_pos.values > 0])
    # df_neg += np.random.uniform(ep/100, ep/10, df_neg.shape)
    # print("... ep: {} => perturbed df_neg:\n{}\n".format(ep, df_neg.head(10)))
    # df_match = pd.concat([df_pos, df_neg])

    # ---------------------------------------------
    # labels = df_match.pop('label')  # this is an 'inplace' operation
    labels = df_match['label']  # labels: a Series
    n_labels = np.unique(labels.values)
    # ---------------------------------------------

    lut = {1: "#3933FF", 0: "#FF3368"} # dict(zip(labels.unique(), "rb"))
    # positive (blue): #3933FF, #3358FF, #e74c3c
    # negative (red) : #FF3368,  #3498db 
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

    if tStandardize:
        X, y, fset, lset = toXY(df_match, cols_y=cols_y, scaler='standardize')
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
    g = sns.clustermap(dfX, figsize=(15, 24.27), z_score=1, row_colors=row_colors, cmap='vlag', metric='cosine', method='complete')  
    # ... need to pass dataframe dfX instead of X! 

    # g.set_xticklabels(g.get_xticklabels(), rotation=45)
    # print(dir(g.ax_heatmap))
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=45)
    # plt.setp(g.ax_heatmap.secondary_yaxis(), rotation=45)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=16, fontweight='light')
    # ... horizontalalignment='right' shifts labels to the left
    
    # colors:  cmap="vlag", cmap="mako", cmap=palette
    # normalization: z_score, stndardize_scale

    output_path = os.path.join(testdir, f'heatmap-{vtype}-{cohort}.png')
    g.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight') 
    # saveFig(plt, output_path, dpi=300)

    ################################################
    df_match = pd.read_csv(input_path, sep=",", header=0, index_col=None, error_bad_lines=False)
    n_samples = 50
    df_match = down_sample(df_match, col_label='label', n_samples=n_samples)
    df_match = relabel(df_match)
    df_pos = df_match[df_match['label']==1]
    df_neg = df_match[df_match['label']==0]
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
    df_match = down_sample(df_match, col_label='label')
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

if __name__ == "__main__": 
    test()