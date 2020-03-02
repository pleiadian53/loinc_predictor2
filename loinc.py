import re, os, random
import pandas as pd
import numpy as np
import data_processor as dp

class SharedProperties(object): 
    col_unknown = "unknown"
    col_joined = "LOINC_MTRT"
    col_corpus = "corpus"

class LoincTable(object): 

    # noisy codes defined in LoincTSet
    # noisy_codes = ['request', 'no loinc needed', '.', ';', 'coumt', 'unloinc']  # made lower case
    
    table_key_map = {'MapTo.csv': ['LOINC', 'MAP_TO'], 
                     'Loinc.csv': 'LOINC_NUM'}

    file_table = 'Loinc.csv'
    input_dir = os.path.join(os.getcwd(), 'LoincTable')
    input_path = os.path.join(input_dir, file_table)

    col_key = col_code = 'LOINC_NUM'
    col_ln = long_name = 'LONG_COMMON_NAME'
    col_sn = short_name = 'SHORTNAME'
    col_com = 'COMPONENT'
    col_sys = 'SYSTEM'
    col_method = 'METHOD_TYP'
    col_prop = 'PROPERTY'

    cols_6p = six_parts = p6 = ['COMPONENT', 'PROPERTY', 'TIME_ASPCT', 'SYSTEM', 'SCALE_TYP', 'METHOD_TYP', ]  # ['CLASS']
    text_cols = ['LONG_COMMON_NAME', 'SHORTNAME', 'RELATEDNAMES2', 'STATUS_TEXT'] + cols_6p

    cols_abbrev = {"LONG_COMMON_NAME": "LN",
                   "SHORTNAME": "SN", 
                   "COMPONENT": "COMP",
                   "PROPERTY": "PROP", 
                   "TIME_ASPCT": "TIME",
                   "SYSTEM": "SYS", 
                   "SCALE_TYP": "SCALE", 
                   "METHOD_TYP": "METHOD"}

    stop_words = ["IN", "FROM", "ON", "OR", "OF", "BY", "AND", "&", "TO", "BY", "", " "]

    @staticmethod
    def load(input_dir='LoincTable', input_file='', **kargs): 
        return load_loinc_table(input_dir=input_dir, input_file=input_file, **kargs)
    @staticmethod
    def load_table(input_dir='LoincTable', input_file='', **kargs):
        return load_loinc_table(input_dir=input_dir, input_file=input_file, **kargs)

class TSet(object): 
    noisy_codes = ['request', 'no loinc needed', '.', ';', 'coumt', 'unloinc'] 
    token_missing = 'unknown'
    token_default = 'unknown'
    token_other = 'other'
    null_codes = [token_other, token_missing]

    codebook={'pos': 1, 'neg': 0, '+': 1, '-': 0}

class LoincTSet(TSet):
    noisy_codes = ['request', 'no loinc needed', '.', ';', 'coumt', 'unloinc'] 
    sites = meta_sender_names = ['Athena', 'Saturn', 'Apollo', 'Poseidon', 'Zeus', 'Plutus', 'Ares']

    col_code = col_target = col_label = 'test_result_loinc_code'
    col_tag = 'medivo_test_result_type'

    # derived features 
    col_corpus = 'corpus'
    col_unknown = 'unknown'
    null_codes = non_codes = [col_unknown, 'other', ]

    cols_result_name_cleaned = ['Site', 'OriginalTestResult', 'CleanedTestResult']
    cols_order_name_cleaned = ["Site", "OriginalTestOrder", "CleanedTestOrder"]
    cols_comments_cleaned = ["Site", "OriginalComments", "CleanedTestComments"]
    cols_specimen_cleaned = ['Site', 'OriginalSpecimen', 'CleanedSpecimen']
    cols_mtrt_cleaned = ["Site", "OriginalMTRT", "CleanedMTRT", ]
    cols_generic = ['site', 'original', 'cleaned']

    cols_abbrev = {'test_order_name': 'TO', 'test_result_name': 'TR', 
        'test_result_comments': 'TC', 
        'medivo_test_result_type': 'MTRT', }

    cols_sdist_matched_loinc = ["PredictedComponent", "ComponentMatchDist", "PredictedSystem", "SystemMatchDist", ]

    file_merged_loinc_mtrt = "loinc_mtrt.corpus"

    @staticmethod
    def get_cols_cleaned(dtype): 
        if dtype == "test_result_name":
            cols = LoincTSet.cols_result_name_cleaned
        elif dtype == "test_order_name":
            cols = LoincTSet.cols_order_name_cleaned 
        elif dtype == 'test_result_comments':
            cols = LoincTSet.cols_comments_cleaned
        elif dtype == 'medivo_test_result_type':
            cols = LoincTSet.cols_mtrt_cleaned
        elif dtype == 'test_specimen_type': 
            cols = LoincTSet.cols_specimen_cleaned
        else: 
            cols = LoincTSet.cols_generic
        return cols
    
    @staticmethod
    def get_sdist_mapped_col_name(dtype, metric='', throw=True):  # the column name for the string-distane features (e.g. string distance between test_order_name and loincmap)
        col = "TestNameMap"
        if dtype == "test_result_name":
            col = "TestResultMap"
        elif dtype == "test_order_name":
            col = "TestOrderMap"
        elif dtype == 'test_result_comments':
            col = "TestCommentsMap"
        elif dtype == 'test_specimen_type': 
            col = "TestSpecimenMap"
        else: 
            msg = f"Unknown data type (i.e. a test-related column in the training data): {dtype}"
            if throw: 
                raise ValueError(msg)
            else: 
                print(msg)
                col = "TestNameMap"
        if metric: col += metric
        return col
    @staticmethod
    def get_sdist_mapped_col_names(dtype, metrics=['LV', 'JW'], throw=True):
        """
        Attributes reserved for re-expressing text values in T-attributes in terms of LOINC vocab.

        """
        cols = []
        for metric in metrics: 
            cols.append( LoincTSet.get_sdist_mapped_col_name(dtype, metric=metric, throw=throw) )
        return cols

    @staticmethod
    def get_sdist_matched_loinc_col_name(dtype, part='Component', vtype='Predicted', metric='LV', throw=True): 
        base = vtype + part
        if not dtype in LoincTSet.cols_abbrev: 
            msg = f"Unknown data type: {dtype}"
            if throw: 
                raise ValueError(msg)
            else: 
                print(msg)

        prefix = LoincTSet.cols_abbrev.get(dtype, '')
        return prefix + base + metric 
    @staticmethod
    def get_sdist_matched_loinc_col_names(dtype, parts=['Component', 'System',], 
           types=['Predicted', 'MatchDist'], metrics=['LV', 'JW'], throw=True): 
        """
        Attributes reserved for T-attributes predicting LOINC parts. 

        """
        
        cols = []
        for part in parts: 
            for metric in metrics: 
                for t in types: 
                    # base = t + part  # PredictedComponent
                    cols.append( LoincTSet.get_sdist_matched_loinc_col_name(dtype, vtype=t, part=part, metric=metric, throw=throw) )
        return cols

    @staticmethod
    def load_tfidf_vars(dtype, **kargs):
        pass

    @staticmethod
    def load_sdist_vars(dtype, **kargs):
        """

        Memo
        ----
        1. sdist variables are generated via 

                feature_gen_sdist.make_string_distance_features()

        2. Examples 
            <file> test_order_name-sdist-vars.csv
            <columns>
 
                test_order_name
                TestOrderMapLV  
                TestOrderMapJW  
                TOPredictedComponentLV  
                TOMatchDistComponentLV  
                TOPredictedComponentJW  
                TOMatchDistComponentJW  
                TOPredictedSystemLV 
                TOMatchDistSystemLV 
                TOPredictedSystemJW 
                TOMatchDistSystemJW
             
                Note: values already cleaned, and standardized (e.g. capitalized)

            other files: 

                test_result_name-sdist-vars.csv

        """
        verbose = kargs.get('verbose', 1)
        throw = kargs.get('throw', False)
        sep = kargs.get('sep', ',')

        input_dir = kargs.get('input_dir', "data") # os.path.join(os.getcwd(), 'result')
        input_file = kargs.get('input_file', f"{dtype}-sdist-vars.csv")  
        input_path = os.path.join(input_dir, input_file)

        df = None
        if os.path.exists(input_path): 
            df = pd.read_csv(input_path, sep=sep, header=0, index_col=None, error_bad_lines=False)
            name = input_file.split('.')[0]
            if verbose: print("(load_sdist_vars) dim(df<{}>): {}\n... columns:\n{}\n".format(name, df.shape, df.columns)) 
        else: 
            msg = "File does not exist at {}\n".format(input_path)
            if throw: 
                raise ValueError(msg)
            else: 
                if verbose: print(msg)
        
        return df
    @staticmethod
    def load_sdist_var_descriptors(target_cols, **kargs):
        kargs['throw'] = False

        adict = {}
        for col in target_cols: 
            df = LoincTSet.load_sdist_vars(col, **kargs)
            if df is not None: 
                adict[col] = df.fillna("")
        return adict

### LoincTSet

# Note: LoincMTRT refactored to loinc_mtrt.py
# class LoincMTRT():
#     pass

class FeatureSet(object):

    cat_cols = ['patient_gender', 
                'patient_state',  # n_uniq=199
                'patient_bill_type',  # n_uniq=31
                'fasting',   # n_uniq=5
                
                'performing_organization_id', # n_uniq=151, m=40%+, NOT part of medivo_test_result_type
                
                'receiving_organization_id', # n_uniq=43, m=50%+, part of medivo_test_result_type
                # 'receiving_organization_name', 
                
                # 'receiving_organization_state', 
                # 'receiving_organization_zip_code', 

                # ---------------------------------------
                # 'ordering_provider_primary_specialty', 
                # 'ordering_provider_secondary_specialty'
                # ---------------------------------------
                
                # 'ordering_practice_lab_account_name',  # high card
                # 'ordering_practice_lab_account_number', # high card
                
                # 'ordering_practice_city', # high card 
                # 'ordering_practice_state', # high card 124? 
                
                # 'ordering_practice_zip_code', # high card,  n_uniq=79392
                # 'ordering_provider_alternate_id_type',   # n_uniq=32
                
                # 'ordering_provider_alternate_id', # n_uniq=132768
                
                # ---------------------------------
                
                'test_result_status', # n_uniq=144
                # 'test_turnaround_time', # n_uniq=417, high missing
                
                'test_order_code',  # n_uniq=27668
                'test_order_name',  # n_uniq=20039
                
                'test_result_code', # n_uniq=23731 (2771052/2891340)
                'test_result_name',  # n_uniq=15581    # <<<< 
                
                'test_result_value',  # n_uniq=35441    # <<<< 
                'test_result_range',   # n_uniq=151, mostly missing   # <<<< 
                
                'test_result_abnormal_flag',  # n_uniq=524, high missing
                
                'test_result_reference_range',  # n_uniq=5735, moderate missing
                
                'test_result_units_of_measure',  # n_uniq=669, m=40%+
                
                # 'test_result_comment_source', # mostly missing
                
                'test_result_comments',  # mostly missing > 80%   # <<<< 
                
                # 'test_priority', 
                # 'test_specimen_collection_volume',
                
                # 'test_specimen_type',  # mostly missing
                
                # 'test_specimen_source', # n_uniq=15971
                # 'test_relevant_clinical_information', # n_uniq=26/
                
                'test_cpt_code',    # n_uniq=655
                
                # 'parent_test_order_code', # n_uniq=5088
                # 'parent_test_order_name', # high missing
                
                # --- datetime ---
                # 'test_specimen_draw_datetime',  # e.g. '2019-08-07T14:47:00.000Z'
                # 'test_specimen_receipt_datetime', #  e.g. '2016-10-06T10:54:00.000Z
                
                # 'test_specimen_analysis_datetime', # high missin
                # 'test_observation_datetime', 
                
                # 'test_observation_reported_datetime', 
                
                'panel_order_code',  # n_uniq=18018
                'panel_order_name',  # n_uniq=11663
                
                # 'parent_panel_order_code', # high missing
                # 'parent_panel_order_name', # high missing
                
                # 'datetime_of_processing',  # no year e.g. 'Jun 29 14:44:25'
                
                # 'meta_ingestion_datetime',
                
                'meta_sender_name',  #  n_uniq=7, m=0% # <<< 
                # ... values: ['Athena' 'Saturn' 'Apollo' 'Poseidon' 'Zeus' 'Plutus' 'Ares']

                #'meta_sender_source',  # n_uniq=2
                # 'meta_sender_type',    # n_uniq=2
                # 'meta_sender_dataset',  # n_uniq=1
                
                'medivo_test_result_type',  # n_uniq=696/(n=67079,N=71224 | r_miss=5.82%) <<<<
            
                ]

    cont_cols = ['age',   # patient_date_of_birth -> age  # <<< 
         ]  

    target_cols = ['test_result_loinc_code', ]
    derived_cols = ['count']  # other possible vars: test result n-th percentile, normalized test frequency

    # cardinality < 100
    low_card_cols = ['patient_gender', 'fasting', 'meta_sender_name' ]
    high_card_cols = list(set(cat_cols)-set(low_card_cols))

    target_columns = cat_cols + cont_cols + target_cols

    @staticmethod
    def get_feature_names(): 
        return FeatureSet.target_columns

    @staticmethod
    def get_targets(): 
        return FeatureSet.target_cols

    @staticmethod
    def join_names(cols, sep='_'):
        return sep.join(sorted(set(cols), key=cols.index))

    @staticmethod
    def join_features(fsets, suffix_set=[]): 
        ns = len(suffix_set)
        if ns > 0: assert ns == len(fsets)
        fsets2 = []
        for i, fset in enumerate(fsets): 
            if len(suffix_set[i]) > 0: 
                fsets2.append(["{}_{}".format(v, suffix_set[i]) for v in fset])
            else: 
                fsets2.append(fset)
        fsets = fsets2
        return np.hstack(fsets)

    @staticmethod
    def to_age(df, col='patient_date_of_birth', new_col='age', add_new_col=True, throw_=False, default_val=None):
        if not col in df.columns: 
            msg = "Error: Missing {}".format(col)
            if throw_: raise ValueError(msg)
                
            # noop
            return df 
        import datetime
        now = datetime.datetime.now()
        
        # date_of_path is rarely NaN but it happens
        if default_val is None: default_val = int(df[col].mean())
        df[col].fillna(value=default_val, inplace=True)
        
        df[new_col] = df[col].apply(lambda x: now.year-int(x))
        if add_new_col: 
            pass
        else: 
            df.drop(col, axis=1, inplace=True)
        return df

### end class FeatureSet

class MatchmakerFeatureSet(FeatureSet): 

    matching_cols = [
        'test_order_name',  # n_uniq=20039
        'test_result_name',  # n_uniq=15581    # <<<< 
        'test_result_range',   # n_uniq=151, mostly missing   # <<<< 
        'test_result_units_of_measure',  # n_uniq=669, m=40%+
        'test_result_reference_range',  # n_uniq=5735, moderate missing
        'test_result_comments',  # mostly missing > 80%   # <<<< 
        'panel_order_name',  # n_uniq=11663
        'medivo_test_result_type',  # n_uniq=696/(n=67079,N=71224 | r_miss=5.82%) <<<<

        # need to also include the label column (y)
        # 'test_result_loinc_code', 

    ]
    # ... these are the candidate features used to match with LOINC descriptors

    cat_cols = [
                # 'patient_gender', 
                # 'patient_state',  # n_uniq=199
                # 'patient_bill_type',  # n_uniq=31
                # 'fasting',   # n_uniq=5
                
                # ---------------------------------
                'test_result_status', # n_uniq=144
                # 'test_turnaround_time', # n_uniq=417, high missing
                
                'test_order_code',  # n_uniq=27668
                
                'test_result_code', # n_uniq=23731 (2771052/2891340)
                'test_result_value',  # n_uniq=35441    # <<<< 
                
                'test_result_abnormal_flag',  # n_uniq=524, high missing
                
                'test_cpt_code',    # n_uniq=655
                
                # 'panel_order_code',  # n_uniq=18018
                
                'meta_sender_name',  #  n_uniq=7, m=0% # <<< 
                # ... values: ['Athena' 'Saturn' 'Apollo' 'Poseidon' 'Zeus' 'Plutus' 'Ares']
            
                ]

    cont_cols = ['age',   # patient_date_of_birth -> age  # <<< 
         ] 

    target_cols = ['test_result_loinc_code', ] # label(s) in the source data
    matching_target_cols = ['label', ]  # the label for the matchmaker dataset

    derived_cols = ['count']  # other possible vars: test result n-th percentile, normalized test frequency

    high_card_cols = list(set(cat_cols)-set(FeatureSet.low_card_cols)-set(matching_cols))

    @staticmethod
    def categorize_features(ts, remove_prefix=True): 
        matching_cols = MatchmakerFeatureSet.matching_cols
        if remove_prefix: 
            new_cols = []
            for i, col in enumerate(matching_cols): 
                new_cols.append(col.replace("test_", ""))
            matching_cols = new_cols
            
        matching_cols = tuple(matching_cols)
        matching_vars = [col for col in ts.columns if col.startswith(matching_cols)]

        target_vars = MatchmakerFeatureSet.matching_target_cols
        derived_vars = MatchmakerFeatureSet.derived_cols

        V = list(ts.columns)
        regular_vars = sorted(set(V)-set(matching_vars)-set(target_vars)-set(derived_vars), key=V.index)

        return (matching_vars, regular_vars, target_vars)

### End class MatchmakerFeatureSet

#########################################################################
# I/O utilities 

def load_loinc_synonyms(input_file='loinc_synonyms.csv', **kargs):
    import LOINCSynonyms as lsyno
    df = lsyno.load_synosyms(input_file=input_file, **kargs)
    if df is None or df.empty: 
        df = lsyno.get_loinc_synonyms()
    return df

def load_loinc_table(input_dir='LoincTable', input_file='', **kargs):
    from transformer import dehyphenate

    sep = kargs.get('sep', ',')
    dehyphen = kargs.get('dehyphenate', False)
    drop_cbit = kargs.get("drop_cbit", False)  # drop correction bit

    if not input_dir: input_dir = kargs.get('input_dir', "LoincTable") # os.path.join(os.getcwd(), 'result')
    if not input_file: input_file = "Loinc.csv"
    input_path = os.path.join(input_dir, input_file)
    assert os.path.exists(input_path), "Invalid path: {}".format(input_path)

    df = pd.read_csv(input_path, sep=sep, header=0, index_col=None, error_bad_lines=False)
    print("> dim(table): {}".format(df.shape)) 

    if dehyphen: 
        col_key = LoincTable.table_key_map.get(input_file, LoincTable.col_code) # 'LOINC_NUM'
        df = dehyphenate(df, col=col_key, drop_cbit=drop_cbit)  # inplace

    return df

# --- LOINC Utilities 
########################################################################

def sample_negatives(code, candidates, n_samples=10, model=None, verbose=1): 
    """
    From the 'candidates' (LOINC codes), choose 'n_samples' codes as negative examples for the target 'code'

    """
    negatives = list(set(candidates)-set([code, ]))
    N = len(negatives)

    if model is None: 
        # random pick N codes that are not the target
        neff = min(N, n_samples)
        if verbose and neff < n_samples: print("(sample_negatives) Not enough candidates (N={}) for sampling n={} negatives".format(N, n_samples))
        negatives = random.sample(negatives, neff)
    else: 
        raise NotImplementedError

    return negatives

def select_samples_by_loinc(df, target_codes, target_cols, **kargs):
    """
    Select training instances from the input data (df) such that: 

    1) the assigned LOINC code of the chosen instance comes from one of the 'target_codes'
    2) selet at most N (n_per_code) instances for each LOINC 
    3) avoid sample duplicates wrt target_cols

    """
    col_code = kargs.get('col_code', LoincTSet.col_code)  # test_result_loinc_code
    n_per_code = kargs.get('n_per_code', 3)
    sizeDict = kargs.get('size_dict', {})  # maps LOINC to (desired) sample sizes

    df = df.loc[df[col_code].isin(target_codes)]

    dfx = []
    for code, dfi in df.groupby([col_code, ]): 
        dfi = dfi.drop_duplicates(subset=target_cols, keep='last')
        n0 = dfi.shape[0]

        n = min(n0, sizeDict[code]) if sizeDict and (code in size_dict) else min(n0, n_per_code)
        dfx.append( dfi.sample(n=n, axis=0) )

    df = pd.concat(dfx, ignore_index=True)

    return df

def expand_by_longname(df, col_src='test_result_loinc_code', 
                col_derived='test_result_loinc_longname', df_ref=None, transformed_vars_only=False, dehyphenate=True):
    # from loinc import LoincMTRT, LoincTable
    # --- LOINC table attributes
    col_ln, col_sn = LoincTable.long_name, LoincTable.short_name
    col_lkey = LoincTable.col_key
    
    if df_ref is None: df_ref = load_loinc_table(dehyphenate=dehyphenate)
    
    # consider the following LOINC codes
    uniq_codes = df[col_src].unique()
    df_ref = df_ref.loc[df_ref[col_lkey].isin(uniq_codes)][[col_lkey, col_ln]]
    
    if transformed_vars_only: 
        return df_ref

    df = pd.merge(df, df_ref, left_on=col_src, right_on=col_lkey, how='left').drop([col_lkey], axis=1)
    df.rename({col_ln: col_derived}, axis=1, inplace=True)

    return df

def sample_loinc_table(codes=[], cols=[], input_dir='LoincTable', input_file='', **kargs): 
    from transformer import dehyphenate
    # from tabulate import tabulate

    col_key = kargs.get('col_key', 'LOINC_NUM')
    n_samples = kargs.get('n_samples', 10) # if -1, show all codes
    verbose = kargs.get('verbose', 1)

    df = load_loinc_table(input_dir=input_dir, input_file=input_file, **kargs)
    df = dehyphenate(df, col=col_key)  # inplace

    if not cols: cols = df.columns.values
    if len(codes) == 0: 
        codes = df.sample(n=n_samples)[col_key].values
    else:  
        codes = np.random.choice(codes, min(n_samples, len(codes)))

    msg = ''
    code_documented = set(df[col_key].values) # note that externally provided codes may not even be in the table!
    adict = {code:{} for code in codes}
    for i, code in enumerate(codes):  # foreach target codes
        ieff = i+1
        if code in code_documented: 
            dfi = df.loc[df[col_key] == code] # there should be only one row for a given code
            assert dfi.shape[0] == 1, "code {} has multiple rows: {}".format(code, tabulate(dfi, headers='keys', tablefmt='psql'))
            msg += f"[{ieff}] loinc: {code}:\n"
            for col in cols: 
                v = list(dfi[col].values)
                if len(v) == 1: v = v[0]
                msg += "  - {}: {}\n".format(col, v)

            adict[code] = sample_df_values(dfi, verbose=0) # sample_df_values() returns a dictionary: column -> value
    if verbose: print(msg)

    return adict


# Comparison methods
#########################################################################
# ... analysis utilties

def compare_6parts(df=None, codes=[], n_samples=-1, cols_6p=[], verbose=1, explicit=True): 
    if df is None: df = load_loinc_table(dehyphenate=True)
    col_code = LoincTable.col_code
    adict = {}
    
    ucodes = df[col_code].unique()
    if len(codes) == 0: 
        n_samples = 20
        codes = np.random.choice(ucodes, min(n_samples, len(ucodes)))
    else: 
        if n_samples > 0: 
            codes = np.random.choice(codes, min(n_samples, len(codes)))

    target_properties = ['SHORTNAME', 'LONG_COMMON_NAME', ]
    if not cols_6p: cols_6p = LoincTable.cols_6p # ['COMPONENT', 'PROPERTY', 'TIME_ASPCT', 'SYSTEM', 'SCALE_TYP', 'METHOD_TYP', ]
    target_properties = target_properties + cols_6p 
    
    for r, row in df.iterrows():
        code = row[col_code]
        
        # assert sum(1 for part in cols_6p if not part in set(row.index)) == 0, "row.index.values={} vs cols(6p): {}".format(
        #     list(row.index.values), cols_6p)

        if len(codes)==0 or (code in codes): 
            if explicit: 
                p6 =  [(part, row[part]) for part in cols_6p]
                six_parts = ""
                for i, part in enumerate(cols_6p): 
                    if i == 0: 
                        six_parts += "{}=\"{}\"".format(part, row[part])
                    else: 
                        six_parts += ", {}=\"{}\"".format(part, row[part])
            else:
                p6 = [row[part] for part in cols_6p]
                six_parts = ": ".join(str(e) for e in p6)

            if verbose: 
                msg = "[{}] {} (6p: {}) =>\n ... {}\n".format(r+1, code, six_parts, row['LONG_COMMON_NAME'])
                print(msg)
                  
            adict[code] = {col: row[col] for col in target_properties}  # (p6, row['LONG_COMMON_NAME'])
    return adict

def compare_short_long_names(df=None, verbose=1, n_display=-1, codes=[], **kargs): 
    verbose = kargs.get('verbose', 1)

    if df is None: df = load_loinc_table(dehyphenate=True)
    col_code = LoincTable.col_code

    ucodes = df[col_code].unique()
    if len(codes) == 0: 
        codes = np.random.choice(ucodes, min(n_display, len(ucodes)))
    else: 
        if n_display > 0: 
            codes = np.random.choice(codes, min(n_display, len(codes)))
    assert n_display < 0 or len(codes) <= n_display

    adict = {}
    for r, row in df.iterrows():
        code = row[col_code]
        long_name = row['LONG_COMMON_NAME']
        short_name = row['SHORTNAME']
        msg = "[{}] {}\n".format(r+1, code)
        msg += "     + {}\n".format(short_name)
        msg += "     + {}\n".format(long_name)
        if verbose: 
            if len(codes)==0 or (code in codes): 
                print(msg)
        adict[code] = (short_name, long_name)
    return adict   

def compare_longname_mtrt(df_mtrt=None, df_loinc=None, n_display=-1, codes=[], **kargs):
    """

    Related 
    -------
    mtrt_to_loinc.demo_parse()
    """
    verbose = kargs.get('verbose', 1)

    table_mtrt = kargs.get('input_mtrt', 'loinc-leela.csv')
    if df_mtrt is None: df_mtrt = LoincMTRT.load_table()
    if df_loinc is None: df_loinc = load_loinc_table(dehyphenate=True)

    col_key_mtrt, col_value_mtrt = LoincMTRT.col_key, LoincMTRT.col_value
    col_key_loinc = LoincTable.col_code
    col_ln, col_sn = 'LONG_COMMON_NAME', 'SHORTNAME'

    # select target LOINC codes to display and analyze
    ucodes = df_mtrt[col_key_mtrt].unique()
    if len(codes) == 0: 
        codes = np.random.choice(ucodes, min(n_display, len(ucodes)))
    else: 
        if n_display > 0: 
            codes = np.random.choice(codes, min(n_display, len(codes)))
    assert n_display < 0 or len(codes) <= n_display

    adict = {}
    for r, row in df_mtrt.iterrows():
        target_code = row[col_key_mtrt] # the LOINC found in MTRT table
        mtrt_name = row[col_value_mtrt]
        assert target_code.find('-') < 0 

        row_loinc = df_loinc.loc[df_loinc[col_key_loinc] == target_code]
        if not row_loinc.empty: 
            assert row_loinc.shape[0] == 1, "Found 1+ matches for code={}:\n{}\n".format(target_code, row_loinc[[col_key_loinc, col_ln]])

            long_name = row_loinc['LONG_COMMON_NAME'].iloc[0] 
            short_name = row_loinc['SHORTNAME'].iloc[0]

            msg = "[{}] {}\n".format(r+1, target_code)
            msg += "    + MTRT: {}\n".format(mtrt_name)
            msg += "    + LN:   {}\n".format(long_name)
            if verbose: 
                if len(codes)==0 or (target_code in codes): 
                    print(msg)
            adict[target_code] = (mtrt_name, long_name)
    return adict 

def compare_by_distance(df=None, df_loinc=None, n_display=-1, codes=[], **kargs): 
    pass

def get_loinc_values(codes, target_cols=[], df_loinc=None, dehyphenate=True):
    from collections import defaultdict

    value_default = ''
    N0 = len(codes)

    # preprocess the codes 
    processed_codes = []
    for code in codes: 
        if pd.isna(code) or len(str(code).strip()) == 0: 
            processed_codes.append(value_default)
        else: 
            if dehyphenate:
                assert code.find('-') < 0
            processed_codes.append(code)
    codes = processed_codes
    #######################

    if df_loinc is None: df_loinc = load_loinc_table(dehyphenate=dehyphenate)
    col_code = LoincTable.col_code
    col_ln, col_sn = 'LONG_COMMON_NAME', 'SHORTNAME'
    if not target_cols: target_cols = [col_ln, col_sn, ]
    df_loinc = df_loinc.loc[df_loinc[col_code].isin(codes)]

    adict = defaultdict(list)
    adict[col_code] = codes
    for code in codes: # foreach input LOINC code
        dfr = df_loinc.loc[df_loinc[col_code]==code]
        if not dfr.empty:  # if we find its descriptor from the LOINC table
            assert dfr.shape[0] == 1
            for col in target_cols: 
                val = dfr[col].iloc[0]
                if pd.isna(val): val = value_default
                adict[col].append(val)
        else: # if this code is not present in the LOINC table
            for col in target_cols: 
                adict[col].append(value_default)
    assert len(adict[target_cols[0]]) == N0, \
        "The number of attribute values (n={}) should match with the number of the codes (n={})".format(len(adict[target_cols[0]]), N0)

    return adict
    
# GROUP-BY methods
########################################################################

def group_by(df_loinc=None, cols=['COMPONENT', 'SYSTEM',], verbose=1, n_samples=-1): 
    if df_loinc is None: df_loinc = load_loinc_table(dehyphenate=True)
    col_code = LoincTable.col_code

    target_properties = ['SHORTNAME', ] # 'LONG_COMMON_NAME'
    cols_6p = LoincTable.cols_6p # ['COMPONENT', 'PROPERTY', 'TIME_ASPCT', 'SYSTEM', 'METHOD_TYP', 'SCALE_TYP']
    target_properties = [col_code, ] + target_properties + LoincTable.cols_6p 

    adict = {}
    for index, dfg in df_loinc.groupby(cols): 
        dfe = dfg[target_properties]
        adict[index] = dfe

    if verbose:
        n_groups = len(adict)
        if n_samples < 0: n_samples = n_groups
        test_cases = set(np.random.choice(range(n_groups), min(n_groups, n_samples)))
        for i, (index, dfg) in enumerate(adict.items()):
            nrow = dfg.shape[0] 
            if i in test_cases and nrow > 1: 
                print("... [{}] => \n{}\n".format(index, dfg.sample(n=min(nrow, 5)).to_string(index=False) ))
    
    return adict

########################################################################

def dehyphenate(df, col='test_result_loinc_code', drop_cbit=False): # 'LOINC_NUM
    """

    Memo
    ----
    1. dropping correction bit leads to more unrecognized codes (not found in the LOINC table), 
       suggesting that hyphenization is a source of noise
    """
    cols = []
    if isinstance(col, str):
        cols.append(col)
    else: 
        assert isinstance(col, (list, tuple, np.ndarray))
        cols = col

    if drop_cbit: 
        for c in cols: 
            df[c] = df[c].replace(regex=r'-\d+', value='')
    else: 
        for c in cols: 
            df[c] = df[c].str.replace('-','')
    return df

def dequote(df, col='Medivo Test Result Type'):
    cols = []
    if isinstance(col, str):
        cols.append(col)
    else: 
        assert isinstance(col, (list, tuple, np.ndarray))
        cols = col

    for c in cols: 
        df[c] = df[c].str.replace('"', '')
    return df

def trim_tail(df, col='test_result_loinc_code', delimit=['.', ';']):
    df[col] = df[col].str.lower().replace('(\.|;)[a-zA-Z0-9]*', '', regex=True)
    return df 

def replace_values(df, values=['.', ], new_value='unknown', col='test_result_loinc_code'):
    for v in values: 
        df[col] = df[col].str.lower().replace(v, new_value)
    
    # ... df[col].str.lower().replace(, 'Unknown') => 'unknown'
    df[col] = df[col].replace(new_value.lower(), new_value) # correction
    return df

def is_canonicalized(df, col_target="test_result_loinc_code", 
        token_default='unknown', token_missing='unknown', token_other='other', 
        target_labels=[], noisy_values=[], columns=[], verbose=1, sample_subset=True):

    if not noisy_values: noisy_values = LoincTSet.noisy_codes # ['request', 'no loinc needed', '.', ';', 'coumt', 'unloinc']

    N = df.shape[0]
    if sample_subset: 
        codes = df[col_target].sample(n=min(N, 100))
    else: 
        codes = df[col_target].values

    for code in codes: 
        if code.find('-') >= 0:
            return False
        if code in noisy_values: 
            return False
        if np.sum(df[col_target].isnull()) != 0: 
            return False

    non_codes = [token_other, token_missing, ]
    Noth = df.loc[df[col_target]==token_other].shape[0]
    Nnan = df.loc[df[col_target]==token_missing].shape[0]
    dfc = df.loc[~df[col_target].isin(non_codes)]

    if len(target_labels) > 0: 
        sc = set(dfc[col_target]) - set(target_labels) 
        if len(sc) > 0: 
            if verbose: print("(is_canonicalized) Found n={} codes not in target set:\n{}\n".format(len(sc), sc))
            return False

    ulabels = dfc[col_target].unique()
    Nc = dfc.shape[0]
    # Nc = N - Noth - Nnan
    if verbose: 
        print("(is_canonicalized) Noth={}, Nnan={}, Nc:{}, Nu: {}".format(Noth, Nnan, Nc, len(ulabels)))

    return True
def canonicalize(df, col_target="test_result_loinc_code", 
        token_default='unknown', token_missing='unknown', token_other='other', 
        target_labels=[], noisy_values=[], columns=[], verbose=1):
    if not noisy_values: noisy_values = LoincTSet.noisy_codes # ['request', 'no loinc needed', '.', ';', 'coumt', 'unloinc']
    
    if verbose: 
        print("(canonicalize) Operations: fill n/a + dehyphenate + replace_values + trim_tail + fill others (non-target classes)")
    
    # fill na 
    df[col_target].fillna(value=token_missing, inplace=True)

    dehyphenate(df, col=col_target)
    replace_values(df, values=noisy_values, new_value=token_missing, col=col_target)
    trim_tail(df, col=col_target, delimit=['.', ';', ])

    df[col_target].replace('', token_missing, inplace=True) 

    # codes that are not in the target set
    if len(target_labels) > 0: 
        if verbose: print("(canonicalize) Focus only on target labels (n={}), labeling the rest as {}".format(len(target_labels), token_other))
        df.loc[~df[col_target].isin(target_labels), col_target] = token_other

    # subset columns (e.g. useful for adding additional data)
    if len(columns) > 0: 
        return df[columns]

    return df

def is_valid_loinc(code, token_default='unknown', dehyphenated=True):
    if code.lower() == token_default: 
        return True
    if dehyphenated: 
        p = re.compile(r'\d+')
    else: 
        p = re.compile(r'\d+(\-\d+)?')

    m = p.match(code)
    if m: 
        return True
    return False

def make_6p_str(df, code, sep='|'):
    return make_6p(df, code, sep=sep, dtype='str')
def make_6p(df, code, col_key='LOINC_NUM', sep='|', dtype='str'): 
    row = df[df[col_key] == code]
    assert row.shape[0] == 1

    # ["COMPONENT","PROPERTY","TIME_ASPCT","SYSTEM","SCALE_TYP","METHOD_TYP"] # ["CLASS"]
    p6 = [row[col].iloc[0] for col in LoincTable.p6]

    if dtype == 'str': 
        return sep.join(str(e) for e in p6)
    elif dtype == 'list':
        return p6 # zip(LoincTable.p6, p6)
    return dict(zip(LoincTable.p6, p6))

def demo_loinc(**kargs):
    from analyzer import load_loinc_table, sample_loinc_table

    df_loinc = load_loinc_table(dehyphenate=True) 
    
    codes = ['10058', '103317']
    for code in codes: 
        s = make_6p(df_loinc, code, dtype='dict')
        print("[{}] {}".format(code, s))

    print("(demo_loinc) Compare LN and SN ...")
    compare_short_long_names(codes=codes)

    print("... Compare LN and MTRT ")
    compare_longname_mtrt(codes=codes)

    pass

def demo_mtrt(): 
    """

    """

    dehyphenate = True
    df_mtrt = LoincMTRT.load_table(dehyphenate=dehyphenate) 

    # --- Loinc to MTRT attributes (from Leela)
    col_mval = LoincMTRT.col_value
    col_mkey = LoincMTRT.col_key    # loinc codes in the mtrt table

    N0 = df_mtrt.shape[0]
    Nu = len(df_mtrt[col_mkey].unique())
    print("(demo) size(df_mtrt): {}, n(unique): {} | E[cols/code]: {}".format(N0, Nu, N0/(Nu+0.0)))

    return

def demo_feature_naming(**kargs):

    dtypes = ['test_result_name', 'test_order_name', ]
    colx = []
    colm = []
    for dtype in dtypes: 
        # T-attributes predicting LOINC parts
        cols = LoincTSet.get_sdist_matched_loinc_col_names(dtype, parts=['Component', 'System',], 
               types=['Predicted', 'MatchDist'], metrics=['LV', 'JW'], throw=True)
        colx.extend(cols)
        print("> predicted | [{}] {}".format(dtype, cols))

        # re-expressed T-attributes
        cols = LoincTSet.get_sdist_mapped_col_names(dtype, metrics=['LV', 'JW'], throw=True)
        colm.extend( cols )
        print("> mapped    | [{}] {}".format(dtype, cols))

    print("(demo) Joining feature sets with suffices that rename these features to prevent from duplicate feature names")
    joined_set = FeatureSet.join_features([['fa', 'fb', 'fc'], ['fb', 'fc', 'fd']], suffix_set=['', 'y', ])
    print("... joined set:\n{}\n".format(joined_set))

    return 

def test(**kargs): 

    # --- LOINC attributes
    # demo_loinc(**kargs)

    # --- MTRT table (leela)
    # demo_mtrt()

    # --- attribute naming 
    demo_feature_naming()

    return

if __name__ == "__main__":
    test()
