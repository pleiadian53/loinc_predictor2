import re, os
import pandas as pd
import numpy as np

class LoincTable(object): 
    cols_6p = six_parts = p6 = ["COMPONENT","PROPERTY","TIME_ASPCT","SYSTEM","SCALE_TYP","METHOD_TYP"]  # ['CLASS']
    text_cols = ['LONG_COMMON_NAME', 'SHORTNAME', 'RELATEDNAMES2', 'STATUS_TEXT']

    # noisy codes 
    noisy_codes = ['request', 'no loinc needed', '.', ';', 'coumt', 'unloinc']  # made lower case
    
    table_key_map = {'MapTo.csv': ['LOINC', 'MAP_TO'], 'Loinc.csv': 'LOINC_NUM'}

    col_key = col_code = 'LOINC_NUM'
    col_key_mtrt = 'test_result_loinc_code'

    file_table = 'Loinc.csv'
    input_dir = os.path.join(os.getcwd(), 'LoincTable')
    input_path = os.path.join(input_dir, file_table)

    long_name = 'LONG_COMMON_NAME'
    short_name = 'SHORTNAME'

    stop_words = ["IN", "FROM", "ON", "OR", "OF", "BY", "AND", "&", "TO", "BY", "", " "]

    @staticmethod
    def load(**kargs): 
        from transformer import dehyphenate
        sep = kargs.get('sep', ',')
        dehyphen = kargs.get('dehyphenate', False)
        input_path = kargs.get('input_path', LoincTable.input_path)
        assert os.path.exists(input_path), "Invalid path: {}".format(input_path)

        df = pd.read_csv(input_path, sep=sep, header=0, index_col=None, error_bad_lines=False)
        print("> dim(table): {}".format(df.shape)) 

        if dehyphen: 
            col_key = LoincTable.table_key_map.get(input_file, LoincTable.col_code) # 'LOINC_NUM'
            df = dehyphenate(df, col=col_key)  # inplace

        return df

class TSet(object): 
    noisy_codes = ['request', 'no loinc needed', '.', ';', 'coumt', 'unloinc'] 
    token_missing = 'unknown'
    token_default = 'unknown'
    token_other = 'other'
    non_codes = [token_other, token_missing]

    codebook={'pos': 1, 'neg': 0, '+': 1, '-': 0}

class LoincTSet(TSet):
    noisy_codes = ['request', 'no loinc needed', '.', ';', 'coumt', 'unloinc'] 
    sites = meta_sender_names = ['Athena', 'Saturn', 'Apollo', 'Poseidon', 'Zeus', 'Plutus', 'Ares']

    col_target = 'test_result_loinc_code'
    col_tag = 'medivo_test_result_type'

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
        
        cols = []
        for part in parts: 
            for metric in metrics: 
                for t in types: 
                    # base = t + part  # PredictedComponent
                    cols.append( LoincTSet.get_sdist_matched_loinc_col_name(dtype, vtype=t, part=part, metric=metric, throw=throw) )
        return cols

### LoincTSet

class LoincMTRT(object):
    header = ['Test Result LOINC Code', 'Medivo Test Result Type']
    col_code = col_key = "Test Result LOINC Code" # use the loinc code as key even though we are primarily interested in predicting loinc from mtrt
    col_value = "Medivo Test Result Type"
    table = 'loinc-leela.csv'
    table_prime = 'loinc-leela-derived.csv'

    stop_words = ["IN", "FROM", "ON", "OR", "OF", "BY", "AND", "&", "TO", "BY", "", " "]

    @staticmethod
    def load_loinc_to_mtrt(input_file=table, **kargs):
        from transformer import dehyphenate
        sep = kargs.get('sep', ',')
        input_dir = kargs.get('input_dir', 'data')
        dehyphen = kargs.get('dehyphenate', True)
        deq = kargs.get('dequote', True)

        df = load_generic(input_file=input_file, sep=sep, input_dir=input_dir) 
        if dehyphen: 
            df = dehyphenate(df, col=LoincMTRT.col_key)  # inplace

        if deq: 
            df = dequote(df, col=LoincMTRT.col_value)
        return df
    @staticmethod
    def save_loinc_to_mtrt(df, **kargs):

        return

    @staticmethod
    def save_derived_loinc_to_mtrt(df, **kargs):
        sep = kargs.get('sep', ',')
        output_dir = kargs.get('output_dir', 'data')
        output_file = kargs.get("output_file", LoincMTRT.table_prime)
        dehyphen = kargs.get('dehyphenate', True)
        deq = kargs.get('dequote', True)

        if dehyphen: 
            df = dehyphenate(df, col=LoincMTRT.col_key)  # inplace
        if deq: 
            df = dequote(df, col=LoincMTRT.col_value)

        save_generic(df, sep=sep, output_file=output_file, output_dir=output_dir) 
        return  
    @staticmethod
    def load_derived_loinc_to_mtrt(**kargs):
        sep = kargs.get('sep', ',')
        input_dir = kargs.get('input_dir', 'data')
        input_file = kargs.get("input_file", LoincMTRT.table_prime)

        # [output[] None if file not found
        return load_generic(input_file=input_file, sep=sep, input_dir=input_dir) 

    @staticmethod
    def transform():
        """
        Transform the LOINC-to-MTRT table (generated from leela) such that the column names 
        are in lowcase with words separated by underscores. 

        e.g. Test Result LOINC Code => test_result_loinc_code

        Additionally, can add additional attributes derived from the baseline columns (e.g. word embedding, 
        term mappings, etc)

        """
        pass

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

    cont_cols = ['age',   # patient_gender -> age  # <<< 
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

### end class FeatureSet

#########################################################################
# I/O utilities 

def load_loinc_table(input_dir='LoincTable', input_file='', **kargs):
    from transformer import dehyphenate
    import loinc as ul

    sep = kargs.get('sep', ',')
    dehyphen = kargs.get('dehyphenate', False)

    if not input_dir: input_dir = kargs.get('input_dir', "LoincTable") # os.path.join(os.getcwd(), 'result')
    if not input_file: input_file = "Loinc.csv"
    input_path = os.path.join(input_dir, input_file)
    assert os.path.exists(input_path), "Invalid path: {}".format(input_path)

    df = pd.read_csv(input_path, sep=sep, header=0, index_col=None, error_bad_lines=False)
    print("> dim(table): {}".format(df.shape)) 

    if dehyphen: 
        col_key = ul.LoincTable.table_key_map.get(input_file, LoincTable.col_code) # 'LOINC_NUM'
        df = dehyphenate(df, col=col_key)  # inplace

    return df

def load_loinc_to_mtrt(input_file='loinc-leela.csv', **kargs):
    from analyzer import load_generic
    sep = kargs.get('sep', ',')
    input_dir = kargs.get('input_dir', os.path.join(os.getcwd(), 'data'))
    df = load_generic(input_dir=input_dir, input_file=input_file, sep=sep) 

    return df


# Comparison methods
#########################################################################
# ... analysis utilties

def compare_6parts(df=None, codes=[], n_samples=-1, cols_6p=[], verbose=1): 
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
    if not cols_6p: 
        cols_6p = ['COMPONENT', 'PROPERTY', 'TIME_ASPCT', 'SYSTEM', 'METHOD_TYP', 'SCALE_TYP']
        target_properties = target_properties + cols_6p 
    
    for r, row in df.iterrows():
        code = row[col_code]
        
        # assert sum(1 for part in cols_6p if not part in set(row.index)) == 0, "row.index.values={} vs cols(6p): {}".format(
        #     list(row.index.values), cols_6p)

        if len(codes)==0 or (code in codes): 
            p6 = [row[part] for part in cols_6p]
            six_parts = ': '.join(str(e) for e in p6)
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
    if df_mtrt is None: df_mtrt = LoincMTRT.load_loinc_to_mtrt(input_file=table_mtrt)
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
    cols_6p = ['COMPONENT', 'PROPERTY', 'TIME_ASPCT', 'SYSTEM', 'METHOD_TYP', 'SCALE_TYP']
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

def dehyphenate(df, col='test_result_loinc_code'): # 'LOINC_NUM'
    cols = []
    if isinstance(col, str):
        cols.append(col)
    else: 
        assert isinstance(col, (list, tuple, np.ndarray))
        cols = col

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

def replace_values(df, values=['.', ], new_value='Unknown', col='test_result_loinc_code'):
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

def save_generic(df, cohort='', dtype='ts', output_file='', sep=',', **kargs):
    output_dir = kargs.get('output_dir', 'data')
    verbose=kargs.get('verbose', 1)
    if not output_file: 
        if cohort: 
            output_file = f"{dtype}-{cohort}.csv" 
        else: 
            output_file = "test.csv"
    output_path = os.path.join(output_dir, output_file)

    df.to_csv(output_path, sep=sep, index=False, header=True)
    if verbose: print("(save_generic) Saved dataframe (dim={}) to:\n{}\n".format(df.shape, output_path))
    return  
def load_generic(cohort='', dtype='ts', input_file='', sep=',', **kargs):
    input_dir = kargs.get('input_dir', 'data')
    verbose=kargs.get('verbose', 1)
    if not input_file: 
        if cohort: 
            input_file = f"{dtype}-{cohort}.csv" 
        else: 
            input_file = "test.csv"
    input_path = os.path.join(input_dir, input_file)

    if os.path.exists(input_path) and os.path.getsize(input_path) > 0: 
        df = pd.read_csv(input_path, sep=sep, header=0, index_col=None, error_bad_lines=False)
        if verbose: print("(load_generic) Loaded dataframe (dim={}) from:\n{}\n".format(df.shape, input_path))
    else: 
        df = None
        if verbose: print("(load_generic) No data found at:\n{}\n".format(input_path))

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

def demo_feature_naming(**kargs):

    dtypes = ['test_result_name', 'test_order_name', ]
    colx = []
    colm = []
    for dtype in dtypes: 
        cols = LoincTSet.get_sdist_matched_loinc_col_names(dtype, parts=['Component', 'System',], 
               types=['Predicted', 'MatchDist'], metrics=['LV', 'JW'], throw=True)
        colx.extend(cols)
        print("> predicted | [{}] {}".format(dtype, cols))

        cols = LoincTSet.get_sdist_mapped_col_names(dtype, metrics=['LV', 'JW'], throw=True)
        colm.extend( cols )
        print("> mapped    | [{}] {}".format(dtype, cols))

def test(**kargs): 

    # --- LOINC attributes
    demo_loinc(**kargs)

    # --- attribute naming 
    demo_feature_naming()

    return

if __name__ == "__main__":
    test()
