import pandas as pd 
import numpy as np
from pandas import DataFrame, Series
import os, re, sys

from tabulate import tabulate
import common
from sklearn.base import BaseEstimator, ClassifierMixin

# local modules 
# from loinc import LoincMTRT as lmt
import loinc
from loinc import LoincMTRT, LoincTable

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
        if not source_table: self.source_table = LoincMTRT.table

        self.table = LoincMTRT.load_loinc_to_mtrt(dehyphenate=True, dequote=True) # input_file/LoincMTRT.table

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
    
    header = LoincMTRT.header  # LoincMTRT
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

def process_text_col(df=None, col='', source_values=[], 
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
        if not col: col = 'processed'
    else: 
        assert df is not None, "Both input dataframe (df) and source values were not given!"
        source_values = df[col].values

    # preprocess source value to ensure that all values are of string type
    source_values = preproces_source_values(source_values=source_values, value_default="")

    if transformed_vars_only: 
        df = DataFrame(source_values, columns=[col, ])  
    else: 
        # noop
        assert df is not None
        df[col] = source_values # overwrite the column values
    
    # preprocess dataframe 
    # token_default = ''
    # df[col] = df[col].fillna(token_default)

    # brackets
    print("(process_text_col) Extracting measurement units (i.e. [...]) ... ")
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
                print("(process_text_col) Weird doc (multiple [])? {}".format(doc))
                # n_malformed += 1
                malformed.append(doc)
                bracketed.append(token_default)
                continue

            bracketed.append( re.search(r'\[(.*?)\]',doc).group(1).strip() )   # use .*? for non-greedy match
        else: 
            bracketed.append(token_default)
    null_rows = set(null_rows)

    # df[col_derived] = df[col].apply(re.search(r'\[(.*?)\]',s).group(1))
    if add_derived: 
        df[col_derived] = bracketed

    dft = df[df[col].str.contains("\[.*?\]")]
    target_index = dft.index
    print("(process_text_col) Malformed []-terms (n={}):\n{}\n".format(len(malformed), display(malformed)))
    print("(process_text_col) After extracting unit (doc type: {}) | n(has []):{}, n(malformed): {} ...\n{}\n".format(docType, 
        dft.shape[0], len(malformed),
        tabulate(dft[cols_target].head(20), headers='keys', tablefmt='psql')))

    # now we don't need []
    if remove_bracket: 
        df[col] = df[col].str.replace("\[.*?\]", '')
        print("(process_text_col) After removing brackets:\n{}\n".format(tabulate(df.iloc[target_index][cols_target].head(20), headers='keys', tablefmt='psql')))

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
    
    print("(process_text_col) Extracting compounds and their abbreviations ... ")
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
                print("(process_text_col) Weird doc (multiple parens)? {}".format(doc))
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
    print("(process_text_col Malformed ()-terms (n={}):\n{}\n".format(len(malformed), display(malformed)))
    print("(process_text_col) After extracting 'compound' & 'abbreviation' (doc type: {}) | n(has_paran):{}, n(malformed):{} ...\n{}\n".format(
        docType, dft.shape[0], len(malformed),
        tabulate(dft[cols_target][[col, col_abbrev]].head(200), headers='keys', tablefmt='psql')))
    # complex cases: 
    #    Hepatitis B virus DNA [log units/volume] (viral load) in Serum or Plasma by NAA with probe detection

    if remove_paran:
        df[col] = df[col].str.replace("\(.*?\)", '')
        print("(process_text_col) After removing parens:\n{}\n".format(tabulate(df.iloc[target_index][cols_target].head(200), headers='keys', tablefmt='psql')))
    # df[col] = new_docs
    ########################################################

    print("(process_text_col) Extracting Postscript ... ")
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
    print("(process_text_col) After extracting additional info (PS) [doc type: {}] | n(has PS): {} ... \n{}\n".format(
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
    col_key = kargs.get('col_key', LoincMTRT.col_key) # 'Test Result LOINC Code'
    save = kargs.get('save', True)
    verbose = kargs.get('verbose', 1)
    sep = kargs.get('sep', ',')
    #######################################
    # --- operational parameters
    add_derived_attributes = kargs.get('add_derived', False)

    if df is None: 
        col_target = ''
        # df = LoincMTRT.load_loinc_to_mtrt(dehyphenate=True, dequote=True) # input_file/LoincMTRT.table

        # default: load source data (training data)
        df = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=True, processed=True)

    # df = process_text_col(df, col='medivo_test_result_type', add_derived=False, save=False)

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

def expand_by_longname(df, col_src='test_result_loinc_code', 
                col_derived='test_result_loinc_longname', df_ref=None, transformed_vars_only=False, dehyphenate=True):
    # from loinc import LoincMTRT, LoincTable
    # --- LOINC table attributes
    col_ln, col_sn = LoincTable.long_name, LoincTable.short_name
    col_lkey = LoincTable.col_key
    ##################################
    # --- Loinc to MTRT attributes (from Leela)
    col_mval = LoincMTRT.col_value
    col_mkey = LoincMTRT.col_key    # loinc codes in the mtrt table
    # df_ref = merge_mtrt_loinc_table(target_cols=[col_lkey, col_ln, col_mval]) # df_mtrt=None, df_loinc=None, dehyphenate=True, target_cols=[]
    
    if df_ref is None: df_ref = loinc.load_loinc_table(dehyphenate=dehyphenate)
    
    # consider the following LOINC codes
    uniq_codes = df[col_src].unique()
    df_ref = df_ref.loc[df_ref[col_lkey].isin(uniq_codes)][[col_lkey, col_ln]]
    
    if transformed_vars_only: 
        return df_ref

    df = pd.merge(df, df_ref, left_on=col_src, right_on=col_lkey, how='left').drop([col_lkey], axis=1)
    df.rename({col_ln: col_derived}, axis=1, inplace=True)

    return df

def merge_mtrt_loinc_table(df_mtrt=None, df_loinc=None, dehyphenate=True, target_cols=[]):
    # from loinc import LoincMTRT, LoincTable

    # --- LOINC table attributes
    col_ln, col_sn = LoincTable.long_name, LoincTable.short_name
    col_lkey = LoincTable.col_key
    ##################################
    # --- Loinc to MTRT attributes (from Leela)
    col_mval = LoincMTRT.col_value
    col_mkey = LoincMTRT.col_key    # loinc codes in the mtrt table

    if df_mtrt is None: df_mtrt = LoincMTRT.load_loinc_to_mtrt(dehyphenate=dehyphenate) 
    if df_loinc is None: df_loinc = loinc.load_loinc_table(dehyphenate=dehyphenate)

    # codeSet = set(df_loinc[col_lkey].values).union(df_mtrt[col_mkey])
    
    df = pd.merge(df_loinc, df_mtrt, left_on=col_lkey, right_on=col_mkey, how='left').drop([col_mkey,], axis=1).fillna('') 
    if len(target_cols) > 0: 
        return df[target_cols]

    return df
def get_corpora_from_merged_loinc_mtrt(df_mtrt=None, df_loinc=None, target_cols=[], dehyphenate=True, sep=" ", remove_dup=True): 
    import transformer as tr
    # from loinc import LoincMTRT, LoincTable

    # --- LOINC table attributes
    col_ln, col_sn = LoincTable.long_name, LoincTable.short_name
    # col_lkey = LoincTable.col_key
    ##################################
    # --- Loinc to MTRT attributes (from Leela)
    col_mval = LoincMTRT.col_value
    # col_mkey = LoincMTRT.col_key    # loinc codes in the mtrt table

    df_merged = merge_mtrt_loinc_table(df_mtrt=df_mtrt, df_loinc=df_loinc, dehyphenate=dehyphenate)

    if not target_cols: target_cols = [col_sn, col_ln, col_mval]
    corpora = tr.conjoin(df_merged, cols=target_cols, transformed_vars_only=True, sep=sep, remove_dup=remove_dup)
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

        dfp = process_text_col(source_values=source_values, col=col_new, add_derived=False, 
                        remove_slot=False,  # remove a given concept from the text? (e.g. if [<unit>] found, then remove [<unit>] from the text)
                        clean=True, standardized=True, save=False, doc_type='training data', transformed_vars_only=True)
        source_values = dfp[col_new].values
    elif df_src is not None: 
        assert len(target_cols) > 0, "Target columns must be specified to extract corpus from a dataframe."
        source_values = np.array([])
        
        # e.g. conjoining test_order_name, test_result_name
        conjoined = tr.conjoin(df, cols=target_cols, transformed_vars_only=True, sep=" ")
        dfp = process_text_col(source_values=conjoined, col=col_new, add_derived=False, 
                        remove_slot=False,  # remove a given concept from the text? (e.g. if [<unit>] found, then remove [<unit>] from the text)
                        clean=True, standardized=True, save=False, doc_type='training data', transformed_vars_only=True)
        # source_values = np.hstack( [source_values, dfp[col_new].values] ) 
        source_values = dfp[col_new].values
    else:  
        # default to use LOINC field and MTRT as the source corpus
        print("(tfidf_pipeline) Use LOINC field and MTRT as the source corpus by default.")

        # B. Using LOINC LN and MTRT as corpus
        df_mtrt = kargs.get('df_mtrt', None)
        df_loinc = kargs.get('df_loinc', None)
        if df_mtrt is None: df_mtrt = LoincMTRT.load_loinc_to_mtrt(dehyphenate=dehyphenate) 
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

        conjoined = get_corpora_from_merged_loinc_mtrt(df_mtrt=df_mtrt[[col_mkey, col_mval]], 
                            df_loinc=df_loinc[[col_lkey, col_sn, col_ln]], target_cols=[col_sn, col_ln, col_mval], 
                               dehyphenate=True, sep=" ", remove_dup=True)
        assert len(conjoined) == df_loinc.shape[0]

        dfp = process_text_col(source_values=conjoined, col=col_new, add_derived=False, 
                        remove_slot=False,  # remove a given concept from the text? (e.g. if [<unit>] found, then remove [<unit>] from the text)
                        clean=True, standardized=True, save=False, doc_type='training data', transformed_vars_only=True)  # transformed_vars_only/True
        print("(model) Processed conjoined loinc LN and MTRT:\n{}\n".format(df_loinc_p.head(30)))


        # [analysis]
        ################################################################
        # ccmap = label_by_performance(cohort='hepatitis-c', categories=['easy', 'hard', 'low'])
        # codes_lsz = ccmap['low']
        # compare_longname_mtrt(df_mtrt=df_mtrt_p, df_loinc=df_loinc_p, codes=codes_lsz)
        ################################################################
        # ... for the most part, they are almost identical

        # use the combined source values as the corpus
        source_values = dfp[col_new].values
    #######################################################
    # ... now we have the source corpus ready

    # preproces_source_values(source_value=source_values, value_default=value_default)
    print("... n={} source values".format( len(source_values) ))

    # model, mydict, corpus = build_tfidf_model(source_values=source_values)
    model = lm.build_tfidf_model(source_values=source_values, standardize=False)
    return model

def gen_sim_features_matching_candidates(codes=[], value_default=0.0):

    # --- LOINC table attributes
    col_ln, col_sn = LoincTable.long_name, LoincTable.short_name
    col_lkey = LoincTable.col_key
    ##################################
    # --- Loinc to MTRT attributes (from Leela)
    col_mval = LoincMTRT.col_value
    col_mkey = LoincMTRT.col_key    # loinc codes in the mtrt table

    df = merge_mtrt_loinc_table() # df_mtrt/None, df_loinc/None, dehyphenate/True, target_cols/[]     
    df = df.loc[df[col_lkey].isin(codes)]

    return 

def gen_sim_features(df=None, target_cols=[], model=None, **kargs): 
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
    from loinc import LoincTable, LoincMTRT, get_loinc_values, load_loinc_table, compare_longname_mtrt
    from analyzer import label_by_performance   # analysis only
    import transformer as tr
    # from scipy.spatial import distance # cosine similarity
    from sklearn.metrics.pairwise import linear_kernel  
    # from transformer import preproces_source_values

    verbose = kargs.get('verbose', 1)
    cohort = kargs.get('cohort', 'hepatitis-c')  # used to index into the desired dataset
    transformed_vars_only = kargs.get('transformed_vars_only', True)
    value_default = kargs.get('value_default', "")
    model = kargs.get('model', None)
    join_target_cols = kargs.get('join_target_cols', False)
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
        conjoined = tr.conjoin(df, cols=target_cols, transformed_vars_only=True, sep=" ")
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
    from loinc import LoincMTRT

    col_key = kargs.get('col_key', LoincMTRT.col_key) # 'Test Result LOINC Code'
    if df is None: df = LoincMTRT.load_loinc_to_mtrt(dehyphenate=True, dequote=True) # input_file/LoincMTRT.table
    
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

    gen_sim_features(df_src=df_src, df_loinc=None, df_map=None, transformed_vars_only=True, verbose=1) 

    return
        
def demo_experta(**kargs):
    from random import choice
    # from experta import *

    return

def demo_loinc_mtrt(): 
    from analyzer import compare_col_values, load_src_data
    cohort = "hepatitis-c"

    df = load_src_data(cohort=cohort, warn_bad_lines=False, canonicalized=True, processed=True)
    dim0 = df.shape
    df = expand_by_longname(df, col_src='test_result_loinc_code', 
           col_derived='test_result_loinc_longname', df_ref=None, transformed_vars_only=False, dehyphenate=True)
    print("(demo) dim0: {} => dim(df): {}, col(df):\n{}\n".format(dim0, df.shape, df.columns.values))

    cols = ['test_result_loinc_longname', 'medivo_test_result_type',]
    compare_col_values(df, cols=cols, n=10, mode='sampling', verbose=1, random_state=53)

    return

def demo_predict(**kargs): 

    predict_by_mtrt()

    return

def test(**kargs): 

    ### Parsing, cleaing, standardizing 
    # demo_parse()
    demo_loinc_mtrt()

    #--- Text feature generation
    # demo_create_tfidf_vars()

    #--- Basic LOINC Prediction
    # demo_predict()

    return

if __name__ == "__main__":
    test()