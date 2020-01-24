import pandas as pd 
import numpy as np
from pandas import DataFrame, Series
import os, re

from tabulate import tabulate
import common
from sklearn.base import BaseEstimator, ClassifierMixin

# local modules 
from loinc import LoincMTRT as lmt


# from utils_plot import saveFig # contains "matplotlib.use('Agg')" which needs to be called before pyplot 
# from matplotlib import pyplot as plt

"""
Rule-based LOINC predictor. 

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

def extract0(mtrt_str, code=''):
    def split_and_strip(s): 
        return ' '.join([str(e).strip() for e in s.split()])
    
    header = lmt.header[0]  # LoincMTRT
    adict = {h:[] for h in header}
    adict[header[0]].append(code)
    adict[header[1]].append(mtrt_str)
    df = DataFrame(adict, columns=adict.keys())

    df = extract(df)
    
    return dict(df.iloc[0])

def extract(df, col='Medivo Test Result Type', save=False):
    """


    Memo
    ----
    1. parenthesis => abbreviation

       hard cases:

            BK virus DNA [Log #/volume] (viral load) in Unspecified specimen by Probe and target amplification method

            fusion transcript  in Blood or Tissue by Fluorescent in situ hybridization (FISH) Narrative
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
            assert e > b
            bracketed.append( re.search(r'\[(.*?)\]',doc).group(1).strip())
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
            assert e > b
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

    # if save: 
    #    # output_file=kargs.get('output_file')
    #    LoincMTRT.save_derived_loinc_to_mtrt(df)

    return df

def tp_clean_and_extract0(df, col='Medivo Test Result Type'):

    # extract paranthesis

    df[col].str.replace(r"\(.*\)","")
     
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
    df = extract(df)

    LoincMTRT.save_derived_loinc_to_mtrt(df)
    
    return

def encode_mtrt_tdidf(docs):  
    """

    Memo
    ----
    1. Medivo Test Result Type
    """

    return



def t_read(**kargs):
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

def t_tdidf(**kargs): 
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
        

def t_predict(**kargs): 

    predict_by_mtrt()

    return

def test(**kargs): 

    ### prediction by loinc-to-mtrt mapping 
    t_predict()

    return

if __name__ == "__main__":
    test()