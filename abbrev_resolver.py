import pandas as pd 
import numpy as np
from pandas import DataFrame, Series
import os, re, collections


def prepare_data(input_dir='data', sep='|', dtype='wikipedia'): 
    """


    Reference 
    ---------
    1. A SIMPLE ALGORITHM FOR IDENTIFYING ABBREVIATION
DEFINITIONS IN BIOMEDICAL TEXT: 

       - https://biotext.berkeley.edu/papers/psb03.pdf

       - https://raw.githubusercontent.com/davidsbatista/lexicons/master/wikipedia-acronyms.txt
    """
    input_path = os.path.join(input_dir, f'{dtype}-acronyms-src.csv')
    # df = pd.read_csv(input_path, sep='\t\t', header=None, index_col=None, error_bad_lines=False, warn_bad_lines=True)
    # ... source is a \t\t separated 

    df = pd.read_csv(input_path, sep='\t', header=None, index_col=None, error_bad_lines=False, warn_bad_lines=True)

    header = ['acronym', 'expansion']
    adict = {h: [] for h in header}
    acronyms = adict['acronym'] = df.iloc[:, 0].values
    fullnames = adict['expansion'] = df.iloc[:, 2].values
    
    df = DataFrame(adict, columns=header)

    output_path =  os.path.join(input_dir, f'{dtype}-acronyms.csv')
    df.to_csv(output_path, sep=sep, index=False, header=True)

    return

def t_acronym_lookup(input_dir='data'):
    delimit = '|'
    # run prepare_data() to generate 'wikipedia-acronyms.csv'
    # prepare_data()
    
    input_path = os.path.join(input_dir, 'wikipedia-acronyms.csv')
    if not os.path.exists(input_path):
        prepare_data()
    # df = pd.read_csv(input_path, sep='\t\t', header=None, index_col=None, error_bad_lines=False, warn_bad_lines=True)
    # ... source is a \t\t separated 

    df = pd.read_csv(input_path, sep=delimit, header=0, index_col=None, error_bad_lines=False, warn_bad_lines=True)

    # header = ['acronym', 'expansion']
    # adict = {h: [] for h in header}
    # acronyms = adict['acronym'] = df.iloc[:, 0].values
    # fullnames = adict['expansion'] = df.iloc[:, 2].values
    
    # adict = dict(zip(df['acronym'].values, df['expansion'].values))
    

    queries = ['CBC', "IGG", "IAT", "CVS", "RBC"]
    for q in queries: 
        fullnames = df[df['acronym'] == q]['expansion'].values
        print("> {} => {}".format(q, fullnames))

    # note: using this "generic" acronym detector, we may not find the desired full names

    return

def test(**kargs): 

    t_acronym_lookup()

    return

if __name__ == "__main__":
    test()