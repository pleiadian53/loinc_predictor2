import json 
import os 
import pandas as pd 
from pandas import DataFrame, Series

# ICD-10-CM CODE, ICD-10-CM CODE DESCRIPTION, CCSR CATEGORY, CCSR CATEGORY DESCRIPTION
"""
1. Find rows whose column match a substring 
   https://davidhamann.de/2017/06/26/pandas-select-elements-by-string/
   
   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.contains.html

"""
inputDir = 'data'
outputDir = 'data'

def load_mapping(input_path='', input_file='', verbose=False):
    basedir = inputDir
    if not input_file: input_file = "DXCCSR-Mapping.csv"
    if not input_path: input_path = os.path.join(basedir, input_file)
    df = pd.read_csv(input_path, sep=',', header=0, index_col=None, error_bad_lines=False)
    
    if verbose: print("(load_mapping) Example:\n{}\n".format(df.head(5)))
    
    return df

def load_categories(input_path='', input_file='', verbose=False):
    basedir = inputDir
    if not input_file: input_file = "DXCCSR-Categories.csv"
    if not input_path: input_path = os.path.join(basedir, input_file)
    df = pd.read_csv(input_path, sep=',', header=0, index_col=None, error_bad_lines=False)
    
    if verbose: print("(load_categories) Example:\n{}\n".format(df.head(5)))
    
    return df 

def search_by_str(condition='hepatitis c', input_path='', input_file='', col="ICD-10-CM CODE DESCRIPTION"):
    df = load_mapping(input_path=input_path, input_file=input_file)
    
    # df = df[df[col].str.match(f".*{condition}[^\w].*", case=False)] # f'{condition}'
    df = df[df[col].str.match(f".*{condition}([^\w.*]|$)", case=False)]
    
    # not working: 
    #    f".*\b{condition}\b.*"
    return df

def name_tod_codes_icd10(**kargs):
    return name_to_codes(**kargs)
def name_to_codes(condition='hepatitis c', input_path='', input_file='', col="ICD-10-CM CODE", verbose=1):
    df = search_by_str(condition=condition, input_path=input_path, input_file=input_file)
    raw_codes = df[col].unique()
    if verbose: 
        if len(raw_codes) == 0: print("(name_to_codes) No codes found matching {}".format(condition))
    # print(f"(string_to_codes) raw codes:\n{raw_codes}\n") 
    
    ncd = 3 # number of digits for the category
    codes = []
    for code in raw_codes: 
        codes.append( code[:ncd] + '.' + code[ncd:] )
    
    return codes

# [todo]
def name_to_codes_icd9(condition='hepatitis c', input_path='', input_file=''):
    if not condition.lower() in ['hepatitis c', 'hepatitis-c', ]: 
        raise NotImplementedError
    # return ["070.41", "070.44", "070.51", "070.54", "070.70", "070.71", "V02.62"]
    return ["07041", "07044", "07051", "07054", "07070", "07071", "V0262"]

def gen_code_set(condition='hepatitis c', to_lower=True, verbose=1):
    # import json
    
    codes_9 = name_to_codes_icd9(condition=condition)
    codes_10 = name_to_codes(condition=condition)
    codes = codes_9 + codes_10
    if to_lower: codes = [code.lower() for code in codes]
        
    if verbose: print("(gen_code_set) condition: {} -> codes:\n{}\n".format(condition, json.dumps(codes)))
    
    return codes

def filter_by_diagnosis(df,  condition='', codes=[], col='diagnosis_codes', col2='billing_diagnosis_codes', verbose=1): 
    
    if condition: 
        codes = gen_code_set(condition=condition, verbose=verbose)
    else: 
        assert len(codes) > 0
    
    indices = []
    for code in codes: 
        for c in [col, col2]: 
            dfm = df[c].str.match(f".*{code}.*", case=False)
            indices.extend(list(dfm.index.values))
    print("(filter_by_diagnosis) Found {} matching rows".format(len(indices)))
    
    # drop rows by indices 
    return df.drop(labels=indices, axis=0)

def query_by_values(values, cols='test_result_loinc_code', exact_match=False):
    """
    Memo
    ----
    1. Find in columns (cols) whose rows match the input values. 
       e.g. Find in 'test_result_loinc_code' all rows having a given set of loinc codes. 
    """
    if isinstance(cols, str): 
        cols = [cols, ]

    if exact_match: 
        n = 0
        for col in cols: 
            for i, val in enumerate(values): 
                if n == 0: 
                    q = f"lower({col}) == {val}"
                else: 
                    q += " " + f"OR lower({col}) == {val}"
                n+=1

    else: 
        for col in cols: 
            for i, val in enumerate(values): 
                if i == 0: 
                    q = f"lower({col}) like '%{val}%'"
                else: 
                    q += " " + f"OR lower({col}) like '%{val}%'"

    return q

def query_by_regex(values, cols='test_result_loinc_code', exact_match=False):
    """
    Generate query statement for .rlike()

    e.g.  ".*kiran.*|.*jay.*"" 

          df.where(col("test_result_loinc_name").rlike("(?i).*kiran.*|.*jay.*")).cache().show()

         

    """
    q = ""
    for i, val in enumerate(values): 
        s = f"{val}"

        if i == 0: 
            q = s  # f"^{val}.*"
        else: 
            q += "|" + s

    return q

def condition_to_codes(condition='hepatitis c', include=['icd9', 'icd10',], to_lower=True):
    codes = []
    # note that the value to condition had better be the exact string representing the condition e.g. hepatitis c but not 'hepatitis-c'

    if 'icd9' in include:
        codes += name_to_codes_icd9(condition=condition)  # [todo]
    if 'icd10' in include or 'icd' in include: 
        codes += name_to_codes(condition=condition)
    if to_lower: codes = [code.lower() for code in codes]

    return codes
     
def gen_query_str(keywords, condition='hepatitis-c', cols=['diagnosis_codes', 'billing_diagnosis_codes' ],  
                   #. col='diagnosis_codes', col2='billing_diagnosis_codes', 
                   # to_lower=True, double_quote=True, 
                   exact_match=False, save=True, **kargs):
    """
    Generate the query string that will select rows whose corresponding 'columns' contain 
    the given 'keywords' (e.g. ICD codes). 
    
    Memo
    ----
    1. diagnosis_codes () example: 
    
       0: ICD9/781.0^ICD9/345.90^ICD9/780.93^ICD9/356.9
       1: ICD10/R73.09^ICD10/M19.90^ICD10/G47.00^ICD10/G20^ICD10/E72.11^ICD10/E03.9
       
    2. query string: 
    
        "lower(diagnosis_codes) like '%g1%' OR ..."
       
    """
    qtype = kargs.get('qtype', 'regex') # the type of query to be executed on Spark Scala 
    vtype = kargs.get('vtype', 'icd')
    verbose = kargs.get('verbose', 1)

    # i) a complete query string: "lower(billing_diagnosis_codes) like '%07071%' OR lower(billing_diagnosis_codes) ... "
    # or 
    # ii) a query string for .rlike(): ".*07071.*|.*g20.* ..."

    # import json
    # codes_9 = name_to_codes_icd9(condition=condition)  # [todo]
    # codes_10 = name_to_codes(condition=condition)
    # codes = codes_9 + codes_10
    # if to_lower: codes = [code.lower() for code in codes]
        
    # print("(gen_query_str) condition: {} -> codes:\n{}\n".format(condition, json.dumps(codes)))
    
    # if exact_match: 
    #     for i, code in enumerate(codes): 
    #         if i == 0: 
    #             q = f"lower({col}) == {code}"
    #         else: 
    #             q += " " + f"OR lower({col}) == {code}"
    #     # secondary attribute (billing_diagnosis_codes), large percentage missing 
    #     for i, code in enumerate(codes):  
    #         q += " " + f"OR lower({col2}) == {code}"
    # else: 
    #     for i, code in enumerate(codes): 
    #         if i == 0: 
    #             q = f"lower({col}) like '%{code}%'"
    #         else: 
    #             q += " " + f"OR lower({col}) like '%{code}%'"
                
    #     # secondary attribute (billing_diagnosis_codes), large percentage missing 
    #     for i, code in enumerate(codes):  
    #         q += " " + f"OR lower({col2}) like '%{code}%'"

    if qtype.startswith('r'): 
        q = query_by_regex(keywords, cols=cols, exact_match=exact_match)
    else: 
        q = query_by_values(keywords, cols=cols, exact_match=exact_match)
    # print("Debug: q=\n{}\n".format(q))
    if save: 
        # save the query to CSV file so that we can resue them on databricks 
        header = ['condition', 'qtype', 'vtype', 'query', ]   # qtype: query type
        adict = {h: [] for h in header}
        adict['query'].append(q)
        adict['condition'].append(condition)
        adict['vtype'].append(vtype)
        adict['qtype'].append(qtype)

        df = DataFrame(adict, columns=header)

        # update the database
        df0 = load_query(cohort=condition)

        if df0 is not None:
            df = df.merge(df0, on=list(df0), how='outer')
            df.drop_duplicates(subset=['condition', 'qtype', 'vtype'], inplace=True, keep='last')

        save_query(df, cohort=condition, verbose=verbose) 
            
    return q

### I/O Utility ### 
def save_query(df, cohort='', output_file='', sep='|', **kargs): 
    from data_processor import save_generic
    save_generic(df, cohort=cohort, dtype='query', sep=sep, **kargs) # output_file/'', 
    return
def load_query(cohort='', output_file='', sep='|', **kargs):
    from data_processor import load_generic
    return load_generic(cohort=cohort, dtype='query', sep=sep, **kargs)


def t_icd(**kargs):
    cohort = "hepatitis-c"
    cohort_canonical = "hepatitis c"
    codes = condition_to_codes(condition=cohort_canonical, include=['icd9', 'icd10',], to_lower=True)
    print("(t_icd) ICD code set:\n{}\n".format(codes))

    q = gen_query_str(keywords=codes, vtype='icd', cols=['diagnosis_codes', 'billing_diagnosis_codes' ])
    print(f"(t_icd) {q}")

    return

def t_loinc(**kargs):
    from analyzer import load_performance
    cohort = 'hepatitis-c'

    df_perf = load_performance(input_dir='result', cohort=cohort)
    n_init = df_perf.shape[0]
    print("> col(pd_perf): {}\n".format(df_perf.columns))

    # q = query_by_regex(df_perf['code'].values)
    q = gen_query_str(keywords=df_perf['code'].values, vtype='loinc', cols='test_result_loinc_code')
    print(f"(t_loinc) {q}")

    return

def test(**kargs): 
    from analyzer import load_performance
    cohort = 'hepatitis-c'
    df_perf = load_performance(input_dir='result', cohort=cohort)
    n_init = df_perf.shape[0]
    print("> col(pd_perf): {}\n".format(df_perf.columns))

    
    # test ICD queries
    t_icd(**kargs)

    # dfp = df_perf.loc[df_perf['mean'] > 0].sort_values(by=['mean', ])

    # Query string using "lower(col) like %g1%" syntax
    # q = query_by_values(df_perf['code'].values)

    # Query string that aggregates all the loinc codes pertaining to a particular cohort (e.g. hepatitis C), which 
    # are then fed into the ML model to get a performance dataframe
    # q = query_by_regex(df_perf['code'].values)

    # test LOINC queries
    t_loinc(**kargs)

    return

if __name__ == "__main__": 
    test()