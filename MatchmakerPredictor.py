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
    df = process_loinc_to_mtrt(df)

    LoincMTRT.save_derived_loinc_to_mtrt(df)

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
        
def demo_experta(**kargs):
    from random import choice
    # from experta import *

    return

def demo_predict(**kargs): 

    predict_by_mtrt()

    return

def test(**kargs): 

    ### prediction by loinc-to-mtrt mapping 
    demo_predict()

    return

if __name__ == "__main__":
    test()