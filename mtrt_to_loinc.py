import pandas as pd 
import numpy as np
from pandas import DataFrame, Series
import os, re, sys

from tabulate import tabulate
import common
from sklearn.base import BaseEstimator, ClassifierMixin

# local modules 
from loinc import LoincMTRT as lmt
from CleanTextData import standardize 
from collections import defaultdict, Counter

from language_model import build_tfidf_model
import config

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
        if not source_table: self.source_table = lmt.table

        self.table = lmt.load_loinc_to_mtrt(dehyphenate=True, dequote=True) # input_file/LoincMTRT.table

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

def process_each(mtrt_str, code=''):
    def split_and_strip(s): 
        return ' '.join([str(e).strip() for e in s.split()])
    
    header = lmt.header  # LoincMTRT
    adict = {h:[] for h in header}
    adict[header[0]].append(str(code))
    adict[header[1]].append(mtrt_str)
    df = DataFrame(adict, columns=adict.keys())

    df = process_loinc_to_mtrt(df)
    
    return dict(df.iloc[0])

def process_loinc_table(): 
    pass

def process_loinc_to_mtrt(df, col='Medivo Test Result Type', save=False, standardized=True, add_derived=True):
    """

    Params
    ------
    * standardized: if True, turn tokens into upper case and apply strip to remove redundant spaces

    Memo
    ----
    1. parenthesis => abbreviation

       hard cases:

            BK virus DNA [Log #/volume] (viral load) in Unspecified specimen by Probe and target amplification method

            fusion transcript  in Blood or Tissue by Fluorescent in situ hybridization (FISH) Narrative

    2. Output dataframe columns: 
       Test Result LOINC Code | Medivo Test Result Type | unit | compound | abbrev | note
    """
    def split_and_strip(s): 
        return ' '.join([str(e).strip() for e in s.split()])

    # brackets 
    cols_target = ["Medivo Test Result Type", "unit"]
    col_derived = 'unit'

    bracketed = []
    token_default = ""
    for doc in df[col].values: 
        b, e = doc.find("["), doc.find("]")
        if b > 0: 
            assert e > b, "weird doc? {}".format(doc)
            bracketed.append( re.search(r'\[(.*?)\]',doc).group(1).strip() )
        else: 
            bracketed.append(token_default)

    # df[col_derived] = df[col].apply(re.search(r'\[(.*?)\]',s).group(1))
    df[col_derived] = bracketed

    dft = df[df[col].str.contains("\[.*\]")]
    print("(extract) after extracting unit:\n{}\n".format(tabulate(dft[cols_target].head(20), headers='keys', tablefmt='psql')))

    # now we don't need []
    df[col] = df[col].str.replace("\[.*\]", '')
    print("(extract) after removing brackets:\n{}\n".format(tabulate(df[cols_target].head(20), headers='keys', tablefmt='psql')))

    ########################################################

    # parenthesis
    abbreviations = []
    compounds = []
    new_docs = []
    col_abbrev = 'abbrev'
    col_comp = 'compound'

    p_abbrev = re.compile(r"(?P<compound>[-a-zA-Z0-9,']+)\s+\((?P<abbrev>.*?)\)")  # p_context
    # ... capture 2,2',3,4,4',5-Hexachlorobiphenyl (PCB)
    # ... cannot capture Bromocresol green (BCG)

    p_aka = re.compile(r"(?P<compound>[-a-zA-Z0-9,']+)/(?P<abbrev>[-a-zA-Z0-9,']+)")
    # ... 3-Hydroxyisobutyrate/Creatinine

    p_abbrev2 = re.compile(r"(?P<compound>([-a-zA-Z0-9,']+)(\s+[-a-zA-Z0-9,']+)*)\s+\((?P<abbrev>.*?)\)")
    # ... von Willebrand factor (vWf) Ag actual/normal in Platelet poor plasma by Immunoassay

    p_by = re.compile(r".*by\s+(?P<compound>([-a-zA-Z0-9,']+)(\s+[-a-zA-Z0-9,']+)*)\s+\((?P<abbrev>.*?)\)")
    p_supplement = p_ps = re.compile(r".*--\s*(?P<ps>([-a-zA-Z0-9,']+)(\s+[-a-zA-Z0-9,']+)*)")
    p_context_by = re.compile(r"by\s+(?P<compound>([-a-zA-Z0-9,']+)(\s+[-a-zA-Z0-9,']+)*)\s+\((?P<abbrev>.*?)\)")
    
    cols_target = ["Medivo Test Result Type", "compound", "abbrev"]
    token_default = ""
    for doc in df[col].values: 
        b, e = doc.find("("), doc.find(")")

        tHasMatch = False
        if b > 0: 
            assert e > b, "weird doc? {}".format(doc)
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

            new_docs.append(doc) # no change
        else:
            new_doc = replaced = re.sub('\(.*\)', '', doc)
            new_docs.append(doc)
            
    df[col_comp] = compounds
    df[col_abbrev] = abbreviations

    dft = df[df[col].str.contains("\(.*\)")]
    print("(extract) After extracting unit:\n{}\n".format(tabulate(dft[cols_target].head(200), headers='keys', tablefmt='psql')))

    # df[col] = df[col].str.replace("\(.*\)", '')
    df[col] = new_docs
    # print("(extract) after removing parens:\n{}\n".format(tabulate(df[cols_target].head(200), headers='keys', tablefmt='psql')))
    ########################################################

    cols_target = ["Medivo Test Result Type", "note"]
    col_derived = 'note'
    token_default = ""
    notes = []
    for doc in df[col].values: 
        m = p_ps.match(doc)
        if m: 
            notes.append(split_and_strip(m.group('ps')))
        else: 
            notes.append(token_default)
    df[col_derived] = notes

    dft = df[df[col].str.contains("--.*")]
    print("(extract) after extracting additional info (PS):\n{}\n".format(tabulate(dft[cols_target].head(50), headers='keys', tablefmt='psql')))
    df[col] = df[col].str.replace("--", '')

    ########################################################
    # remove extra spaces
    df[col] = df[col].apply(split_and_strip)

    if save: 
        # output_file=kargs.get('output_file')
        LoincMTRT.save_derived_loinc_to_mtrt(df)


    # columns: Test Result LOINC Code, Medivo Test Result Type, unit, compound, abbrev, note
    return df

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

def process_tag(df=None, col='medivo_test_result_type', source_values=[], 
                   add_derived=True, remove_slot=True, clean=True, standardized=True, save=False, 
                   transformed_vars_only=True, **kargs): 
    """
    Parse long strings, such as LOINC's long name field and MTRT, which typically serve as "tags"

    Note that the name of the slots/derived attributes may not be always what they meant to be used. 
    e.g. 'unit' typically refers to measurement units but sometimes other kinds of values could be enclosed within brackets as well. 

    """
    def split_and_strip(s): 
        return ' '.join([str(e).strip() for e in s.split()])

    from CleanTextData import clean_term, standardize

    # for debugging and testing only
    docType = kargs.get('doc_type', "long name")
    remove_bracket = kargs.get('remove_bracket', True)
    remove_paran = kargs.get('remove_paran', True)

    if len(source_values) > 0: 
        transformed_vars_only = True
    else: 
        assert df is not None, "Both input dataframe (df) and source values were not given!"
        source_values = df[col].values

    # preprocess source value to ensure that all values are of string type
    source_values_processed = []
    n_null = n_numeric = 0
    token_default = ''
    for source_value in source_values: 
        if pd.isna(source_value): 
            source_values_processed.append(token_default)
            n_null += 1
        elif isinstance(source_value, (int, float, )): 
            n_numeric += 1
            source_values_processed.append(str(source_value))
        else: 
            source_values_processed.append(source_value.strip())
    source_values = source_values_processed

    if transformed_vars_only: 
        df = DataFrame(source_values, columns=[col, ])  
    else: 
        # noop
        df[col] = source_values # fill the processed source values 
    
    # preprocess dataframe 
    # token_default = ''
    # df[col] = df[col].fillna(token_default)

    # brackets
    print("(process_tag) Extracting measurement units (i.e. [...]) ... ")
    cols_target = [col, "unit"]
    col_derived = 'unit'
    bracketed = []
    token_default = ""
    null_rows = []
    # n_malformed = 0
    malformed = []
    for r, doc in enumerate(df[col].values): 
        if pd.isna(doc) or len(str(doc)) == 0: 
            null_rows.append(r)
            bracketed.append(token_default)
            continue

        b, e = doc.find("["), doc.find("]")
        if b > 0: 
            if not e > b: 
                print("(process_tag) Weird doc (multiple [])? {}".format(doc))
                # n_malformed += 1
                malformed.append(doc)
                bracketed.append(token_default)
                continue

            bracketed.append( re.search(r'\[(.*?)\]',doc).group(1).strip() )   # use .*? for non-greedy match
        else: 
            bracketed.append(token_default)
    null_rows = set(null_rows)

    # df[col_derived] = df[col].apply(re.search(r'\[(.*?)\]',s).group(1))
    if add_derived: df[col_derived] = bracketed

    dft = df[df[col].str.contains("\[.*?\]")]
    target_index = dft.index
    print("(process_tag)  Malformed []-terms (n={}):\n{}\n".format(len(malformed), display(malformed)))
    print("(process_tag) After extracting unit (doc type: {}) | n(has []):{}, n(malformed): {} ...\n{}\n".format(docType, 
        dft.shape[0], len(malformed),
        tabulate(dft[cols_target].head(20), headers='keys', tablefmt='psql')))

    # now we don't need []
    if remove_bracket: 
        df[col] = df[col].str.replace("\[.*?\]", '')
        print("(process_tag) after removing brackets:\n{}\n".format(tabulate(df.iloc[target_index][cols_target].head(20), headers='keys', tablefmt='psql')))

    ########################################################

    # parenthesis
    abbreviations = []
    compounds = []
    new_docs = []
    col_abbrev = 'abbrev'
    col_comp = 'compound'

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
    
    print("(process_tag) Extracting compounds and their abbreviations ... ")
    cols_target = [col, "compound", "abbrev"]
    token_default = ""
    n_null = n_malformed = 0
    malformed = []
    for r, doc in enumerate(df[col].values): 
        # if r in null_rows: 
        #     abbreviations.append(token_default)
        #     compounds.append(token_default)
        #     continue

        b, e = doc.find("("), doc.find(")")
        tHasMatch = False
        if b > 0: 
            if not (e > b): 
                print("(process_tag) Weird doc (multiple parens)? {}".format(doc))
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
            if remove_paran: 
                doc = re.sub('\(.*?\)', '', doc)
 
                b, e = doc.find("("), doc.find(")")
                assert b < 0 or e < 0, "Multiple parans? {}".format(doc)

        new_docs.append(doc)
            
    if add_derived: 
        df[col_comp] = compounds
        df[col_abbrev] = abbreviations

    dft = df[df[col].str.contains("\(.*?\)")]
    target_index = dft.index
    print("(process_tag Malformed ()-terms (n={}):\n{}\n".format(len(malformed), display(malformed)))
    print("(process_tag) After extracting 'compound' & 'abbreviation' (doc type: {}) | n(has_paran):{}, n(malformed):{} ...\n{}\n".format(
        docType, dft.shape[0], len(malformed),
        tabulate(dft[cols_target][[col, col_abbrev]].head(200), headers='keys', tablefmt='psql')))
    # complex cases: 
    #    Hepatitis B virus DNA [log units/volume] (viral load) in Serum or Plasma by NAA with probe detection

    if remove_paran:
        df[col] = df[col].str.replace("\(.*?\)", '')
        print("(extract) After removing parens:\n{}\n".format(tabulate(df.iloc[target_index][cols_target].head(200), headers='keys', tablefmt='psql')))
    # df[col] = new_docs
    ########################################################

    print("(process_tag) Extracting Postscript ... ")
    cols_target = [col, "note"]
    col_derived = 'note'
    token_default = ""
    notes = []
    for r, doc in enumerate(df[col].values): 
        # if r in null_rows: 
        #     notes.append(token_default)
        #     continue

        m = p_ps.match(doc)
        if m: 
            notes.append(split_and_strip(m.group('ps')))
        else: 
            notes.append(token_default)
    df[col_derived] = notes

    dft = df[df[col].str.contains("--.*")]
    print("(process_tag) After extracting additional info (PS) [doc type: {}] | n(has PS): {} ... \n{}\n".format(
        docType, dft.shape[0], tabulate(dft[cols_target].head(50), headers='keys', tablefmt='psql')))

    if remove_slot: 
        df[col] = df[col].str.replace("--", " ")

    ########################################################

    if standardized: # this has to come before clean operation
        df[col] = df[col].apply(standardize)

    # clean text 
    if clean: 
        # dft = df.loc[df[col].str.contains(r'\bby\b', flags=re.IGNORECASE)]
        # index_keyword = dft.index
        # print("... Prior to cleaning | df(by):\n{}\n".format(dft.head(10)))

        df[col] = df[col].apply(clean_term)
        
        # dft = df.loc[df[col].str.contains(r'\bby\b', flags=re.IGNORECASE)]
        # assert dft.empty, "... not cleaned:\n{}\n".format(df.iloc[index_keyword].head(10))
        # assert dft.empty, "... not cleaned:\n{}\n".format(dft.head(20))

    else: 
        # remove extra spaces
        df[col] = df[col].apply(split_and_strip)

    # if save: 
    #     output_file=kargs.get('output_file')
    #     LoincMTRT.save_derived_loinc_to_mtrt(df)
    return df

def clean_mtrt(df=None, col_target='medivo_test_result_type', **kargs): 
    """

    Related
    -------
    MapLOINCFields.parse_loinc()
    """
    from CleanTestsAndSpecimens import clean_terms, clean_term
    from loinc import LoincMTRT

    siteWordCount = defaultdict(Counter)
    mtrtList = defaultdict(list)
    medivo_test_result_type = config.tagged_col

    #######################################
    # --- generic parameters
    cohort = kargs.get('cohort', 'hepatitis-c')    
    col_key = kargs.get('col_key', lmt.col_key) # 'Test Result LOINC Code'
    save = kargs.get('save', True)
    verbose = kargs.get('verbose', 1)
    sep = kargs.get('sep', ',')
    #######################################
    # --- operational parameters
    add_derived_attributes = kargs.get('add_derived', False)

    if df is None: 
        col_target = ''
        # df = lmt.load_loinc_to_mtrt(dehyphenate=True, dequote=True) # input_file/LoincMTRT.table

        # default: load source data (training data)
        df = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=True, processed=True)

    # df = process_tag(df, col='medivo_test_result_type', add_derived=False, save=False)

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

def parallel_features_loinc_vs_mtrt(df_src=None, df_loinc=None, df_mtrt=None, **kargs): 
    """
    Use source texts to train a TD-IDF model

    i) MTRT: leela 
    ii) LOINC: LOINC table

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
    from loinc import LoincTable, LoincMTRT, get_loinc_values, load_loinc_table, compare_longname_mtrt
    from analyzer import label_by_performance   # analysis only

    verbose = kargs.get('verbose', 1)
    transformed_vars_only = kargs.get('transformed_vars_only', 1)
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

    if df_src is None: 
        # load the original data, so that we can use the punctuation info to extract concepts (e.g. measurements are specified in brackets)
        isProcessed = False
        df_src = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=True, processed=isProcessed)

    target_cols = [col_code, col_mtrt, ]
    assert np.all([col in df_src.columns for col in target_cols]), "Missing some columns (any of {}) in the input".format(target_cols)
    ############################################################################
    # ... source representation
    mtrts = df_src[col_mtrt].values
    input_mtrt = process_tag(source_values=mtrts, col='', add_derived=True, remove_slot=False, 
                            clean=True, standardized=True, save=False, doc_type='training data (MTRT)') # transformed_vars_only/True 

    codes = df_src[col_code].values
    loinc_table = get_loinc_values(codes, target_cols=[col_ln, col_sn, ], df_loinc=None, dehyphenate=True)
    input_loinc = process_tag(source_values=loinc_table[col_ln], add_derived=True, 
                            remove_slot=False,  # remove a given concept from the text? (e.g. if [<unit>] found, then remove [<unit>] from the text)
                            clean=True, standardized=True, save=False, doc_type='training data (LN)') # transformed_vars_only/True
    print("(parallel) Size of input: {} =?= {}".format(len(input_mtrt), len(input_loinc)))
    print("(parallel) Processed input data loinc LNs:\n{}\n".format(input_loinc.head(30)))
    print("(parallel) Processed input MTRTs:\n{}\n".format(input_mtrt.head(30)))
    ############################################################################

    header = ['d_tfidf_loinc_to_mtrt', 'd_jw_unit', 'd_jw_compound']
    if transformed_vars_only: 
        # columns(derived df_mtrt): Test Result LOINC Code, Medivo Test Result Type, unit, compound, abbrev, note
        df_output = DataFrame(columns=header)
    else: 
        df_incr = DataFrame(columns=header)
        df_output = pd.concat([df_src, df_incr], axis=1)

    col_key = kargs.get('col_key', lmt.col_key) # 'Test Result LOINC Code'
    if df_mtrt is None: df_mtrt = LoincMTRT.load_loinc_to_mtrt(dehyphenate=True) 
    if df_loinc is None: df_loinc = load_loinc_table(dehyphenate=True)

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
    df_loinc = df_loinc[[col_lkey, col_ln, col_sn]]
    df_mtrt = df_mtrt[[col_mkey, col_mval]] 

    df_loinc_p = process_tag(df=df_loinc, col=col_ln, add_derived=True, 
                               remove_slot=False,  # remove a given concept from the text? (e.g. if [<unit>] found, then remove [<unit>] from the text)
                               clean=True, standardized=True, save=False, doc_type='loinc table (LN)', transformed_vars_only=False)  # transformed_vars_only/True
    print("(parallel) Processed loinc LNs:\n{}\n".format(df_loinc_p.head(30)))
    
    # ... processed df_loinc 
    df_mtrt_p = process_tag(df=df_mtrt, col=col_mval, add_derived=True, 
                            remove_slot=False, 
                            clean=True, standardized=True, save=False, doc_type='leela mapping (MTRT)', transformed_vars_only=False)
    print("(parallel) Processed MTRTs:\n{}\n".format(df_mtrt_p.head(30)))

    # [analysis]
    ################################################################
    # ccmap = label_by_performance(cohort='hepatitis-c', categories=['easy', 'hard', 'low'])
    # codes_lsz = ccmap['low']
    # compare_longname_mtrt(df=df_mtrt_p, df_loinc=df_loinc_p, codes=codes_lsz)
    ################################################################
    # ... for the most part, they are almost identical

    # use the combined source values as the corpus
    n_offset = df_loinc_p.shape[0]
    source_values = np.hstack( [ df_loinc_p[col_ln].values, df_mtrt_p[col_mval].values] )
    print("... n={} source values".format( len(source_values) ))
    model, mydict, corpus = build_tfidf_model(source_values=source_values)

    # Show the TF-IDF weights
    # for doc in model[corpus]:
    #     print([[mydict[id_], np.around(freq, decimals=2)] for id_, freq in doc])
    
    # --- Compute d_tfidf_loinc_to_mtrt
    print("... n(mydict): {}, size(corpus): {}".format(len(mydict), len(corpus)))
    # ... n(mydict) ~ # of unique tokens in the corpus
    #     size(corpus) ~ size(source_values)

    vector = model[corpus[0]]


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
    from loinc import LoincMTRT

    col_key = kargs.get('col_key', lmt.col_key) # 'Test Result LOINC Code'
    if df is None: df = lmt.load_loinc_to_mtrt(dehyphenate=True, dequote=True) # input_file/LoincMTRT.table
    
    # verify 
    for code in df['Test Result LOINC Code'].values: 
        assert code.find('-') < 0
    for v in df['Medivo Test Result Type'].values: 
        assert v.find('"') < 0
    assert len(mtrt_str) > 0 or code is not None

    
    print("(predict_by_mtrt) df.columns: {}".format(df.columns.values))
    print("(predict_by_mtrt) dim(df): {} | n_codes".format(df.shape, len(df[col_key].unique())) )
    
    # string matching algorithm
    df = process_loinc_to_mtrt(df, save=True)
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

def demo_create_tfidf_vars(**kargs):
    from analyzer import label_by_performance, col_values_by_codes, load_src_data

    cohort = "hepatitis-c"
    col_target = 'test_result_loinc_code'
    categories = ['easy', 'hard', 'low']  # low: low sample size
    ccmap = label_by_performance(cohort='hepatitis-c', categories=categories)

    codes_lsz = ccmap['low']
    print("(demo) n_codes(low sample size): {}".format(len(codes_lsz)))
    codes_hard = ccmap['hard']
    print("...    n_codes(hard): {}".format(len(codes_hard)))
    target_codes = list(set(np.hstack([codes_hard, codes_lsz])))
    dfp = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=True, processed=True)
    # adict = col_values_by_codes(target_codes, df=dfp, cols=['test_result_name', 'test_order_name'], mode='raw')
    df_src = dfp.loc[dfp[col_target].isin(target_codes)]
    print("(demo) dim(input): {}".format(df_src.shape))

    parallel_features_loinc_vs_mtrt(df_src=df_src, df_loinc=None, df_map=None, transformed_vars_only=True, verbose=1) 

    return
        
def demo_experta(**kargs):
    from random import choice
    # from experta import *

    return

def demo_predict(**kargs): 

    predict_by_mtrt()

    return

def test(**kargs): 

    ### Parsing, cleaing, standardizing 
    # demo_parse()

    #--- Text feature generation
    demo_create_tfidf_vars()

    #--- Basic LOINC Prediction
    # demo_predict()

    return

if __name__ == "__main__":
    test()