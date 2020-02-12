
# coding: utf-8

# In[1]:

import pandas as pd
from pandas import DataFrame
import numpy as np
import os, sys
import time
from collections import defaultdict
import csv
import seaborn as sb
# import rpy2.robjects as robjects
# from APISearchRequests import *
# import rpy2.robjects.numpy2ri as numpy2ri
# from rpy2.robjects.packages import importr

### Local Modules
import config

from MapLOINCFields import *
from CleanTextData import *
from tabulate import tabulate

import transformer
from loinc import LoincTSet
from utils_sys import highlight

# ### Read in aggregate data & join with parsed/cleaned test name and specimen

def build_cube():
    """

    Memo
    ----
    1. modified 
    """

    def get_path(dtype='test_result_name'): 
        return os.path.join(config.out_dir, "cleaned_{}.csv".format(dtype))

    if config.print_status == 'Y':
        print('Building analytic data cube')
    agg_source_data = pd.read_csv(config.in_file, sep=config.delim, quoting=csv.QUOTE_NONE, 
              encoding = "ISO-8859-1", keep_default_na=False, na_values=config.missing)
    
    # check file availability (run CleanTextData.py) 
    tDataCleaned = True
    for dtype in ['test_result_name', 'test_order_name', 'test_specimen_type', ]: 
        if not os.path.exists(os.path.join(config.out_dir, "cleaned_{}.csv".format(dtype))): 
            tDataCleaned = False
            break

    if tDataCleaned:

        # cleaned_tests
        cleaned_test_results = pd.read_csv(get_path(dtype='test_result_name'), sep="|", quoting=csv.QUOTE_NONE,
                                encoding = "ISO-8859-1", keep_default_na=False, na_values=config.missing)
 
        cleaned_test_orders = pd.read_csv(get_path(dtype='test_order_name'), sep="|", quoting=csv.QUOTE_NONE,
                                encoding = "ISO-8859-1", keep_default_na=False, na_values=config.missing)

        cleaned_specimen = pd.read_csv(get_path(dtype='test_specimen_type'), sep="|", quoting=csv.QUOTE_NONE,
                                   encoding = "ISO-8859-1", keep_default_na=False,
                                  na_values=config.missing)
    else:
        mapped  = import_source_data()
        cleaned_tests, cleaned_specimen = mapped['test_order_names'], mapped['test_specimen_type']
    
    agg_source_data[config.test_col] = agg_source_data[config.test_col].str.strip().str.upper()
    agg_source_data[config.spec_col] = agg_source_data[config.spec_col].str.strip().str.upper()
    agg_source_data[config.units] = agg_source_data[config.units].str.strip().str.upper()

    if agg_source_data[config.site].dtypes != cleaned_specimen['Site'].dtypes:
        agg_source_data[config.site] = agg_source_data[config.site].astype(str)
        cleaned_specimen.Site = cleaned_specimen.Site.astype(str)
        cleaned_tests.Site = cleaned_tests.Site.astype(str)
    
    ################################################################################
    cleaned_test = cleaned_test_orders
    joined_dat = agg_source_data.merge(cleaned_tests, how='left', left_on=[config.site, config.test_col], 
                                   right_on=['Site', 'OriginalTestResult'])  # OriginalTestName
    joined_dat = joined_dat.merge(cleaned_specimen, how='left', left_on=[config.site, config.spec_col], 
                                 right_on=['Site', 'OriginalSpecimen'])
    joined_dat = joined_dat.drop(['Site_x', 'Site_y', 'OriginalTestName', 'OriginalSpecimen'], axis=1)
    joined_dat = joined_dat[(~(joined_dat[config.test_col].isnull())) & (~(joined_dat[config.spec_col].isnull()))]

    ## Get total # of lab results per site, create normalized 'FreqPercent' variable
    joined_dat = joined_dat.merge(pd.Series.to_frame(joined_dat.groupby(config.site)[config.count].sum(), name='TotalCount').reset_index(),
                                 how='inner', left_on=config.site, right_on=config.site)
    joined_dat['FreqPercent'] = joined_dat[config.count]/joined_dat.TotalCount * 100.0
    
    return joined_dat


# ### Read in UMLS mapped names and put in data frame

def compile_cuis(data):
    master_list = defaultdict(list)
    for i in range(data.shape[0]):
        if data.at[i, 'SourceTerm'] not in master_list:
            master_list[data.at[i, 'SourceTerm']].append(data.at[i, 'CUI'])
        if data.at[i, 'CUI'] not in master_list[data.at[i, 'SourceTerm']]:
            master_list[data.at[i, 'SourceTerm']].append(data.at[i, 'CUI'])
    return master_list


def add_cuis_to_cube(dat):
    if config.print_status == 'Y':
        print('Adding UMLS CUIs')
    feature_col_number = config.num_cuis
    if (os.path.exists(config.out_dir + "UMLS_Mapped_Specimen_Names.csv") and 
        os.path.exists(config.out_dir + "UMLS_Mapped_Test_Names.csv")):
        master_spec_UMLS = compile_cuis(pd.read_csv(config.out_dir + "UMLS_Mapped_Specimen_Names.csv", sep="|"))
        master_test_UMLS = compile_cuis(pd.read_csv(config.out_dir + "UMLS_Mapped_Test_Names.csv", sep="|"))
    else:
        test_input, specimen_input = data_setup()
        master_spec_UMLS = compile_cuis(parse_dat(specimen_input, "Specimen"))
        master_test_UMLS = compile_cuis(parse_dat(test_input, "Test"))
        
    for i in range(feature_col_number):
        dat['SpecCUI{0}'.format(i + 1)] = 'NONE'
        dat['TestCUI{0}'.format(i + 1)] = 'NONE'
        
    for j in range(dat.shape[0]):
        for k in range(len(master_spec_UMLS[dat.at[j, 'CleanedSpecimen']])):
            if k < feature_col_number:
                dat.at[j, 'SpecCUI{0}'.format(k + 1)] = master_spec_UMLS[dat.at[j, 'CleanedSpecimen']][k]
        for l in range(len(master_test_UMLS[dat.at[j, 'CleanedTestName']])):
            if l < feature_col_number:
                dat.at[j, 'TestCUI{0}'.format(l + 1)] = master_test_UMLS[dat.at[j, 'CleanedTestName']][l]
    return dat


# ### Map LOINC System tokens to LOINC Long Name Tokens

# short_to_long, parsed_loinc_fields = parse_loinc()

def map_loinc_system(parsed_loinc_fields=None, sep="|"):
    """

    Memo
    ----
    To handle abbreviations contained in the LOINC System field,
    we used string distance matching with the Jaro-Winkler metric
    to find the corresponding words with the smallest edit distance in the
    LOINC Long Name field. We mapped the System token to the resulting
    distance-matched Long Name token and/or acronym expansion
    """
    from MapLOINCFields import parse_loinc
    from pyjarowinkler import distance

    if config.print_status == 'Y':
        print('Mapping LOINC System')

    if parsed_loinc_fields is None: 
        _, parsed_loinc_fields = parse_loinc()

    fpath = os.path.join(config.out_dir, "LOINC_System_to_Long.csv")  # input_path
    if os.path.exists(fpath):
        system_map = pd.read_csv(fpath, sep=sep)
    else:
        # numpy2ri.activate()
        # stringdist = importr('stringdist', lib_loc=config.lib_loc)

        loinc_syst = parsed_loinc_fields[['System', 'LongName']]
        loinc_syst = loinc_syst[(~pd.isnull(loinc_syst.System)) & (loinc_syst.System != '')].reset_index(drop=True)
        loinc_syst.System = loinc_syst.System.str.split(" ")
        loinc_syst.LongName = loinc_syst.LongName.str.split(" ")
        
        system_tokens = pd.Series([y for x in loinc_syst.System for y in x]).unique()
        longname_tokens = pd.Series([y for x in loinc_syst.LongName for y in x]).unique()
        
        system_df = pd.DataFrame(0, index=system_tokens, columns=longname_tokens)
        print("(map_loinc_system) dim(system_df): {} ... n(system token) vs n(LN token)".format(system_df.shape))

        n_rows = loinc_syst.shape[0]
        for i in range(n_rows):  # foreach row in system LN table 
            for j, term in enumerate(loinc_syst.System[i]):  # foreach System token 

                # term vs tokens in long name
                # dists = stringdist.stringdist(term, loinc_syst.LongName[i], method = 'jw', p=0)
                dists = [1.0-distance.get_jaro_distance(term, token, winkler=True, scaling=0.1) for token in loinc_syst.LongName[i]]
                # ... distance.get_jaro_distance returns "similarity"
                
                bestMatch = loinc_syst.LongName[i][np.argmin(dists)]   # e.g. BLD vs BLOOD
                system_df.loc[term, bestMatch] = system_df.loc[term, bestMatch] + 1

                # i.e. after looping through all rows (seeing all system tokens), we will keep track of ...
                # ... how many times S<i> been "best-matched-by-count" to LN<j>

        # [test]
        print("(map_loinc_system) Counts @ System token matching LN:\n{}\n".format(system_df.head(100)))
        # ... sparse matrix: system token vs LN token, each cell -> count of (best) match

        high_count = system_df.idxmax(axis=1).values   # numpy array
        # find token index with the highest counts for each system token

        system_map = pd.DataFrame({'SystemToken': system_tokens, 'SystemMap': high_count})
        # ... system token (s<i>) vs the count 
        if config.write_file_loinc_parsed:
            # output_path = os.path.join(config.out_dir, "LOINC_System_to_Long.csv")
            system_map.to_csv(fpath, sep=sep, index=False)

    return system_map

# ### Get highest count, longest string for each short name token mapped by counts to long name tokens

def map_loinc_token_counts(short_to_long=None):
    """
    Find out the extension of the corresponding acronyms by performing cross walk between SN and LN
    Choose the longest match with highest count

    Memo
    ----
    1. for i, dfe in short_to_long.groupby(['Token']):
           print("... [{}] => \n{}\n".format(i, dfe.head(5))) 

    ... [DISEASE] =>
          Token                TokenMap  Count
119892  DISEASE                SYMPTOMS      2
119893  DISEASE       SYMPTOMS DISEASES      2
119894  DISEASE  SYMPTOMS DISEASES FIND      2
119895  DISEASE                 DISEASE     61
119896  DISEASE          DISEASE CANCER      1

... [DIPYRIDAMOLE] =>
              Token                       TokenMap  Count
83986  DIPYRIDAMOLE                   DIPYRIDAMOLE      4
83987  DIPYRIDAMOLE             DIPYRIDAMOLE SERUM      1
83988  DIPYRIDAMOLE      DIPYRIDAMOLE SERUM PLASMA      1
83989  DIPYRIDAMOLE           DIPYRIDAMOLE INDUCED      2
83990  DIPYRIDAMOLE  DIPYRIDAMOLE INDUCED PLATELET      2

    2. transform is an operation used in conjunction with groupby

    """

    if short_to_long is None: 
        short_to_long, _ = parse_loinc()

    ## Get highest count for each short name token mapped by counts to long name tokens (from the MapLOINCFields)
    idx = short_to_long.groupby(['Token'])['Count'].transform(max) == short_to_long['Count']

    # group by 
    # short_to_long: 'Token' 'TokenMap', 'Count'
    ####################################################
    # for i, dfe in short_to_long.groupby(['Token']): 
    #     print("... [{}] => \n{}\n".format(i, dfe.head(5)))

    ####################################################
    # print(short_to_long.groupby(['Token'])['Count'].transform(max))  # take the max count for each mapping in each row
    
    # sys.exit(0)

    ##############################

    loinc_terms_max = short_to_long[idx].drop('Count', 1).reset_index(drop = True)
    loinc_terms_max['FinalTokenMap'] = np.nan
    ## If TokenMap contains elongation of TokenAbbreviation, populate AcronymMap column
    loinc_terms_max['AcronymnMap'] = np.nan
    for i in range(loinc_terms_max.shape[0]):
        if not pd.isnull(loinc_terms_max.TokenMap[i]):
            token_map_tokens = loinc_terms_max.TokenMap[i].split(" ")

            ################
            # loinc_terms_max.TokenMap[i] => 
            # PRONOUNCE QUIP => ['PRONOUNCE', 'QUIP']
            # VP6 => ['VP6']
            # SYNCHRONOUS TUMOR => ['SYNCHRONOUS', 'TUMOR']
            # MARGIN INVOLVEMENT COLORECTAL => ['MARGIN', 'INVOLVEMENT', 'COLORECTAL']
            # EXTRAMURAL VEIN INVASION => ['EXTRAMURAL', 'VEIN', 'INVASION']
            # GREATER THAN 10MM => ['GREATER', 'THAN', '10MM']
            ################

            if len(loinc_terms_max.Token[i]) > 1 and len(token_map_tokens) >= len(loinc_terms_max.Token[i]):
                # DAT             | DIRECT ANTIGLOBULIN TEST          |             nan | DIRECT ANTIGLOBULIN TEST
                # len(DAT) > 1 and ( len(DIRECT ANTIGLOBULIN TEST) = 3 >= len(DAT) == 3)

                counter = 0
                string = ""
                for j in range(len(loinc_terms_max.Token[i])):
                    if loinc_terms_max.Token[i][j:j+1] == token_map_tokens[j][0:1]:  # check first char against that of the token
                        counter = counter + 1
                        if len(string) < 1:
                            string = token_map_tokens[j]  # use the first word as a start
                        else:
                            string = string + " " + token_map_tokens[j]
                # ... now finish the check if the match is actually an acronym

                if counter == len(loinc_terms_max.Token[i]):
                    loinc_terms_max.loc[i, 'AcronymnMap'] = string
    return loinc_terms_max

def group_func(group):
    ## If Token == TokenMap, make this the FinalTokenMap, otherwise use the shortest TokenMap as the key
    if not group['AcronymnMap'].isnull().all():
        _ = group['FinalTokenMap'].fillna(str(group.AcronymnMap.dropna().unique()[0]), inplace=True)
    elif group['Token'].astype(str).any() == group['TokenMap'].astype(str).any():
        # print("(group_func) group:\n{}\n".format(group))  # a dataframe (of the given group)
        # print("(group_func) group['FinalTokenMap'] <- group['Token'] ~\n{} <- {}".format(group['FinalTokenMap'], group['Token']))
        _ = group['FinalTokenMap'] = group['Token']
    elif not group['SystemMap'].isnull().all():
        _ = group['FinalTokenMap'] = group['SystemMap']
    else:
        _ = group['FinalTokenMap'] = min(group.TokenMap, key=len)
    return group


def combine_loinc_mapping(verbose=1, save=True, **kargs):
    from MapLOINCFields import parse_loinc

    short_to_long, parsed_loinc_fields = parse_loinc()
    system_map_final = map_loinc_system(parsed_loinc_fields) # loinc system (abbrev) to its full name via comparison with LN
    loinc_terms_max = map_loinc_token_counts(short_to_long)  # LOINC SN to LN, best match by counts
    if verbose: 
        print("(combine_loinc_mapping) system token mapping to 'best' LN token (by count):\n{}\n".format(
            tabulate(system_map_final.head(20), headers='keys', tablefmt='psql') ))
        # e.g. 
        # SystemToken     | SystemMap
        # PLAS            | PLASMA          |
        # BPU             | BLOOD           |

        print("(combine_loinc_mapping) loinc_terms_max:\n{}\n".format( 
            tabulate(loinc_terms_max.head(100), headers='keys', tablefmt='psql') ))
        # e.g. 
        # Token           | TokenMap                          |   FinalTokenMap | AcronymnMap
        # IAT             | INDIRECT ANTIGLOBULIN TEST        |             nan | INDIRECT ANTIGLOBULIN TEST
        # RBC             | RED BLOOD CELLS                   |             nan | RED BLOOD CELLS

    loincmap = system_map_final.merge(loinc_terms_max, how='outer', left_on='SystemToken', right_on='Token')

    # print("... loinc_map (1):\n{}\n".format( tabulate(loincmap.head(200), headers='keys', tablefmt='psql') ))

    loincmap.loc[loincmap.Token.isnull(), 'Token'] = loincmap.loc[loincmap.Token.isnull(), 'SystemToken']
    loincmap.loc[loincmap.TokenMap.isnull(), 'TokenMap'] = loincmap.loc[loincmap.TokenMap.isnull(), 'SystemMap']

    # print("... loinc_map (2):\n{}\n".format( tabulate(loincmap.head(200), headers='keys', tablefmt='psql') ))
    # +-----+-----------------+-----------------+-----------------+-------------------------------+-----------------+-----------------------+
    # |     | SystemToken     | SystemMap       | Token           | TokenMap                      |   FinalTokenMap | AcronymnMap           |
    # |-----+-----------------+-----------------+-----------------+-------------------------------+-----------------+-----------------------|
    # |   0 | HEART           | LEFT            | HEART           | HEART                         |             nan | nan                   |
    # |   1 | SER             | SERUM           | SER             | AB                            |             nan | nan                   |
    # |   2 | PLAS            | PLASMA          | PLAS            | POST                          |             nan | nan                   |
    # |   3 | BPU             | BLOOD           | BPU             | BLOOD                         |             nan | nan                   |
    # |   4 | DONOR           | DONOR           | DONOR           | DONOR                         |             nan | nan                   |
    # |   5 | RBC             | RED             | RBC             | RED                           |             nan | nan                   |
    # |   6 | RBC             | RED             | RBC             | RED BLOOD                     |             nan | nan                   |
    # |   7 | RBC             | RED             | RBC             | RED BLOOD CELLS               |             nan | RED BLOOD CELLS       |
    # ... 
    # | 181 | CSF             | CEREBRAL        | CSF             | CEREBRAL SPINAL FLUID         |             nan | CEREBRAL SPINAL FLUID |

    if config.print_status == 'Y':
        print('Generating LOINC Groups')
    loincmap = loincmap.groupby('Token').apply(group_func)

    # [test]
    # print("... loinc_map (3) groupby token")
    # for i, dfe in loincmap.groupby(['Token']): 
    #     print("... [{}] => \n{}\n".format(i, dfe.head(5)))

    # e.g. 
    # [ZIMELIDINE] =>
    #       SystemToken SystemMap       Token                 TokenMap FinalTokenMap AcronymnMap
    # 18308         NaN       NaN  ZIMELIDINE               ZIMELIDINE    ZIMELIDINE         NaN
    # 18309         NaN       NaN  ZIMELIDINE         ZIMELIDINE SERUM    ZIMELIDINE         NaN
    # 18310         NaN       NaN  ZIMELIDINE  ZIMELIDINE SERUM PLASMA    ZIMELIDINE         NaN

    # [YMDD] =>
    #       SystemToken SystemMap Token      TokenMap FinalTokenMap AcronymnMap
    # 18919         NaN       NaN  YMDD             B             B         NaN
    # 18920         NaN       NaN  YMDD       B VIRUS             B         NaN
    # 18921         NaN       NaN  YMDD  B VIRUS YMDD             B         NaN

    loincmap = loincmap[['Token', 'FinalTokenMap']].drop_duplicates().reset_index(drop=True)

    # [output]
    # e.g. acronym cases, abbrev cases, trival cases
    # Token            | FinalTokenMap
    # -----------------------------------------
    # RBC              | RED BLOOD CELLS
    # SER              | SERUM
    # CSF              | CEREBRAL SPINAL FLUID
    # -----------------------------------------
    # PLAS             | PLASMA                
    # BPU              | BLOOD     
    # -----------------------------------------            
    # WRIST            | WRIST                 
    # LYMPHATIC        | LYMPHATIC

    # final test 
    test_cases = ['YMMD', 'SPELLS', 'HEART', 'PLAS', 'CSF', ]
    print("(combine_loinc_mapping) Final map examples:\n{}\n".format(loincmap.loc[loincmap['Token'].isin(test_cases)]))

    if save: save_loincmap(loincmap)

    return loincmap, short_to_long, parsed_loinc_fields

def load_loincmap(input_dir='data', input_file='', **kargs):
    cohort = kargs.get('cohort', 'generic')
    sep = kargs.get('sep', ',')
    verbose = kargs.get('verbose', 1)
    exception_ = kargs.get('exception_', False)

    if not input_dir: input_dir = os.path.join(os.getcwd(), input_dir) # os.path.join(os.getcwd(), 'result')
    if not input_file: input_file = f"loincmap-{cohort}.csv" 
    input_path = os.path.join(input_dir, input_file)
    if os.path.exists(input_path): 
        df = pd.read_csv(input_path, sep=sep, header=0, index_col=None, error_bad_lines=False)
        if verbose: 
            print("> dim(loincmap): {}".format(df.shape)) 
    else: 
        msg = "Invalid path: {}\n".format(input_path)
        if exception_: 
            raise ValueError(msg)
        else: 
            print(msg)

        df = None

    return df

def save_loincmap(df, output_dir='data', output_file='', **kargs): 
    cohort = kargs.get('cohort', 'hepatitis-c')
    sep = kargs.get('sep', ',')
    verbose = kargs.get('verbose', 1)
    n_display = kargs.get('n_display', 200)

    output_dir = kargs.get('output_dir', os.path.join(os.getcwd(), output_dir)) 
    output_file = f"loincmap-{cohort}.csv" 
    output_path = os.path.join(output_dir, output_file)
    df.to_csv(output_path, sep=sep, index=False, header=True)

    if verbose: 
        print('(save) Saving loincmap dataframe to:\n{}\n ... #'.format(output_path))
        for i, (token, final_token) in enumerate(zip(df['Token'], df['FinalTokenMap'])):
            if i > n_display: break
            print(f"{token} -> {final_token}")
    return

def load_match_matrix(input_file='', metric='JW', **kargs):
    cohort = kargs.get('cohort', 'generic')
    sep = kargs.get('sep', ',')
    verbose = kargs.get('verbose', 1)
    exception_ = kargs.get('exception_', False)

    input_dir = kargs.get('input_dir', 'data') # os.path.join(os.getcwd(), 'result')
    if not input_file: 
        if metric: 
            input_file = f"match_matrix-{metric}-{cohort}.csv"  # test tokens vs loinc tokens
        else: 
            input_file = f"match_matrix-{cohort}.csv" 
    input_path = os.path.join(input_dir, input_file)

    if os.path.exists(input_path): 
        df = pd.read_csv(input_path, sep=sep, header=0, index_col=None, error_bad_lines=False)
        if verbose: 
            print("> dim(match_matrix): {}".format(df.shape)) 
    else: 
        msg = "Invalid path: {}\n".format(input_path)
        if exception_: 
            raise ValueError(msg)
        else: 
            print(msg)

        df = None
    return df 
def save_match_matrix(df, output_file='', metric='JW', **kargs):
    cohort = kargs.get('cohort', 'hepatitis-c')
    sep = kargs.get('sep', ',')
    verbose = kargs.get('verbose', 1)
    # n_display = kargs.get('n_display', 200)

    output_dir = kargs.get('output_dir', 'data') 
    if not output_file: 
        if metric: 
            output_file = f"match_matrix-{metric}-{cohort}.csv"  # test tokens vs loinc tokens
        else: 
            output_file = f"match_matrix-{cohort}.csv" 
    output_path = os.path.join(output_dir, output_file)
    df.to_csv(output_path, sep=sep, index=False, header=True)
    return

def distance_jaro_winkler(x, y, verbose=0): 
    from pyjarowinkler import distance
    d = 1.0
    # value_default = "unknown"
    try: 
        d = 1.0-distance.get_jaro_distance(x, y, winkler=True, scaling=0.1)
    except Exception as e: 
        if verbose: print(e)
        if pd.isna(x): x = ""
        if pd.isna(y): y = ""
        x = str(x)
        y = str(y)
        if x == "" and y == "": 
            d = 0.0   # as if they are a perfect match 
        elif "" in (x, y):  # one of them is empty
            d = 1.0   # as if they are the least match
        else: 
            if verbose: print("(distance_jaro_winkler) Possibly non-string values (x: {}, y:{})".format(x, y))
            d = 1.0-distance.get_jaro_distance(x, y, winkler=True, scaling=0.1)
    return d

def distance_jaro_winkler_agg(s1, s2, sep=" ", verbose=0): 
    d = 1.0
    if x == "" and y == "": 
        d = 0.0 
    elif "" in (s1, s2): 
        d = 1.0
    else: 
        token_to_string_distances = []
        s1_tokenized = s1.split(sep)
        s2_tokenized = s2.split(sep)

        # [todo]
    return d
        
# Find best string matches between source data test or specimen tokens and the loincmap (i.e. 
# the mapping rom LOINC short name to LOINC long name words)
def get_matches(data_col, loincmap, save=False):
    """
    
    Params
    ------
    save: set to True to save the output dataframe (the best "translated" version of test strings in terms of LOINC vocab)
    """
    def resolve_by_first_char(src, mapped): 
        if src[0] != mapped[0]: # if the fist character doesn't even match, then don't used the mapped token 
            return src
        return mapped
    def resolve(src, mapped):
        
        # add other rules here
        final_token = resolve_by_first_char(src, mapped)

        return final_token

    import stringdist
    from pyjarowinkler import distance

    if config.print_status == 'Y':
        print('String Distance Matching Source Data Terms to LOINC')
    # numpy2ri.activate()
    # stringdist = importr('stringdist', lib_loc=config.lib_loc)

    # data_col: a list/array of strings from test-related attributes (e.g. test_order_name)
    tokenized_list = [str(data_col[k]).split() for k in range(len(data_col))]
    longest_phrase = len(max(tokenized_list, key=len))

    match_matrix_LV = pd.DataFrame(np.nan, index=data_col, columns=range(longest_phrase))
    match_matrix_JW = pd.DataFrame(np.nan, index=data_col, columns=range(longest_phrase))
    rows = len(tokenized_list)
    for i in range(rows):   # foreach test string value (e.g. test_result_name: "Mitochondria M2 Ab")
        if config.print_status == 'Y' and i % 500 == 0:
            print('Matching Term', i, '/', rows)
        for j in range(len(tokenized_list[i])):   # foreach token (e.g. Mitochondria)
            if not pd.isnull(tokenized_list[i][j]):
                term = str(tokenized_list[i][j])  # a token in the attribute text
                
                # dists_LV = stringdist.stringdist(tokenized_list[i][j], loincmap.Token.values, method='lv')
                dists_LV = [stringdist.levenshtein(term, str(token)) for token in loincmap.Token.values]
                # ... the distance between each test_* token to  
                
                # dists_JW = stringdist.stringdist(tokenized_list[i][j], loincmap.Token.values, method='jw', p=0)
                dists_JW = [distance_jaro_winkler(term, str(token)) for token in loincmap.Token.values]
                # ... 'scaling' should not exceed 0.25, otherwise the similarity could become larger than 1

                # but if we can't find a match, the keep the original token? 

                mappedLV = loincmap.FinalTokenMap[np.argmin(dists_LV)] 
                match_matrix_LV.iloc[i, j] = resolve(term, mappedLV)  # use short name as an anchor to compare (test_result_name and long name)
                
                mappedJW = loincmap.FinalTokenMap[np.argmin(dists_JW)]
                match_matrix_JW.iloc[i, j] = resolve(term, mappedJW)

    if save: 
        save_match_matrix(match_matrix_LV, metric='LV')
        save_match_matrix(match_matrix_JW, metric='JW')
        
    ############
    # ... each token in the T-strings (values of test_result_name, etc.) is mapped to a token from the loincmap wrt to a given string distance
    # > match_matrix_LV
    # 
    # HEPATITIS C ANTIBODY SUPPLEMENTAL TESTING           HEPATITIS           C       ANTIBODY  SUPPLEMENTAL  TESTING  NaN  
    # SUBOXONE TOTAL URINE                                 NALOXONE         IGE          URINE           NaN      NaN  NaN  
    # AMMONIA                                               AMMONIA         NaN            NaN           NaN      NaN  NaN 
    # 
    ############
    # > match_matrix_JW
    # 
    # HEPATITIS C ANTIBODY SUPPLEMENTAL TESTING           HEPATITIS           C       ANTIBODY  SUPPLEMENTAL  TESTING  NaN  
    # SUBOXONE TOTAL URINE                                SUBSTANCE         IGE          URINE           NaN      NaN  NaN  
    # AMMONIA                                               AMMONIA         NaN            NaN           NaN      NaN  NaN  
    # HEMOGLOBINOPATHY EVAL                      HEMOGLOBINOPATHIES  EVALUATION            NaN           NaN      NaN  NaN  
    # HERPES SIMPLEX IGG                                     HERPES      HERPES            IGG           NaN      NaN  NaN  
    # ...                                                       ...         ...            ...           ...      ...  ...  
    # PROTIME                                             PROTAMINE         NaN            NaN           NaN      NaN  NaN  
    # ALK PHOS ISOENZYME                                         PH   PHOSPHATE         ENZYME           NaN      NaN  NaN  
    # JAK2 V617F QUAL RFX EXON 12                              JAK2           P  QUALIFICATION          BETA     GENE    V  
    # CREATININE CLEARANCE                               CREATININE   CLEARANCE            NaN           NaN      NaN  NaN  
    # TREPONEMA PALLIDUM ANTIBODIES                       TREPONEMA    PALLIDUM     ANTIBODIES           NaN      NaN  NaN

    return match_matrix_LV, match_matrix_JW


def concatenate_match_results0(input_matrix, dataType):
    n_rows = input_matrix.shape[0]
    n_cols = input_matrix.shape[1]
    for i in range(n_rows):
        for j in range(1, n_cols):
            if not pd.isnull(input_matrix.iloc[i, j]):
                input_matrix.iloc[i, 0] = input_matrix.iloc[i, 0] + " " + input_matrix.iloc[i, j]
    if dataType == 1:
        return pd.DataFrame(input_matrix.iloc[:, 0].values, index=input_matrix.index, columns=['TestNameMap'])
    else:
        return pd.DataFrame(input_matrix.iloc[:, 0].values, index=input_matrix.index, columns=['SpecimenMap'])

def remove_duplicates(s, sep=" "):
    tokens = str(s).split(sep)
    return sep.join(sorted(set(tokens), key=tokens.index))

def concatenate_match_results(input_matrix, dataType, metric='JW', remove_dup=True):
    n_rows = input_matrix.shape[0]
    n_cols = input_matrix.shape[1]
    for i in range(n_rows):
        for j in range(1, n_cols):
            if not pd.isnull(input_matrix.iloc[i, j]):
                input_matrix.iloc[i, 0] = input_matrix.iloc[i, 0] + " " + input_matrix.iloc[i, j]
        
        if remove_dup: 
            if isinstance(input_matrix.iloc[i, 0], str): 
                input_matrix.iloc[i, 0] = remove_duplicates(input_matrix.iloc[i, 0])

    col = LoincTSet.get_sdist_mapped_col_name(dataType, metric=metric)
    return pd.DataFrame(input_matrix.iloc[:, 0].values, index=input_matrix.index, columns=[col])

def add_string_distance_features():
    joined_data = build_cube()
    data = add_cuis_to_cube(joined_data)
    
    loincmap, *_ = combine_loinc_mapping()
    
    unique_tests = data[~data.CleanedTestName.isnull()].CleanedTestName.unique()
    unique_specimen_types = data[~data.CleanedSpecimen.isnull()].CleanedSpecimen.unique()

    # matching unique test names with loinc long names
    test_match_matrix_LV, test_match_matrix_JW = get_matches(unique_tests, loincmap)
    spec_match_matrix_LV, spec_match_matrix_JW = get_matches(unique_specimen_types, loincmap)

    if config.print_status == 'Y': print('Concatenating String Match Results')

    concat_lv_test_match_result = concatenate_match_results0(test_match_matrix_LV, 1)
    concat_jw_test_match_result = concatenate_match_results0(test_match_matrix_JW, 1)
    concat_lv_spec_match_result = concatenate_match_results0(spec_match_matrix_LV, 2)
    concat_jw_spec_match_result = concatenate_match_results0(spec_match_matrix_JW, 2)

    concat_lv_test_match_result.columns.values[0] = 'TestNameMapLV'
    concat_jw_test_match_result.columns.values[0] = 'TestNameMapJW'
    concat_lv_spec_match_result.columns.values[0] = 'SpecimenMapLV'
    concat_jw_spec_match_result.columns.values[0] = 'SpecimenMapJW'

    concat_test_match_result = pd.concat([concat_lv_test_match_result, concat_jw_test_match_result], axis=1)
    concat_spec_match_result = pd.concat([concat_lv_spec_match_result, concat_jw_spec_match_result], axis=1)
    
    dat = data.merge(concat_test_match_result, how='left', left_on='CleanedTestName', right_index=True)
    dat = dat.merge(concat_spec_match_result, how='left', left_on='CleanedSpecimen', right_index=True)
    
    loinc_comp_syst = parsed_loinc_fields[['LOINC', 'Component', 'System']]
    loinc_comp_syst = loinc_comp_syst[(~pd.isnull(loinc_comp_syst.System)) & 
        (loinc_comp_syst.System != '')].reset_index(drop=True)
    loinc_comp_syst['ExpandedSystem'] = np.nan
    loinc_comp_syst.ExpandedSystem = loinc_comp_syst.ExpandedSystem.astype(object)
    
    loinc_num_set = loinc_comp_syst.LOINC.unique()

    if config.print_status == 'Y':
        print('Generating LOINC System Field Expansion')
    rows = loinc_comp_syst.shape[0]
    for i in range(rows):
        if config.print_status == 'Y' and i % 5000 == 0:
            print('Row', i, '/', rows)
        if not pd.isnull(loinc_comp_syst.System[i]):
            loinc_comp_syst.at[i, 'System'] = loinc_comp_syst.System[i].split(" ")
            for term in loinc_comp_syst.System[i]:
                mapped_term = loincmap.loc[loincmap.Token == term, 'FinalTokenMap'].values[0]
                if pd.isnull(loinc_comp_syst.ExpandedSystem[i]):
                    loinc_comp_syst.at[i, 'ExpandedSystem'] = mapped_term
                else:
                    loinc_comp_syst.at[i, 'ExpandedSystem'] = loinc_comp_syst.ExpandedSystem[i] + " " + mapped_term

    unique_combos = dat[['TestNameMapJW', 'SpecimenMapJW', 'TestNameMapLV', 'SpecimenMapLV']].drop_duplicates().reset_index(drop=True)
    unique_components = loinc_comp_syst.Component.unique()
    unique_system = loinc_comp_syst[~pd.isnull(loinc_comp_syst.ExpandedSystem)].ExpandedSystem.unique()
    
    unique_combos = pd.concat([unique_combos, pd.DataFrame(columns=['PredictedComponentJW', 'ComponentMatchDistJW', 'PredictedComponentLV', 'ComponentMatchDistLV', 
               'PredictedSystemJW', 'SystemMatchDistJW', 'PredictedSystemLV', 'SystemMatchDistLV'])], sort=False)
    
    numpy2ri.activate()
    stringdist = importr('stringdist', lib_loc=config.lib_loc)

    if config.print_status == 'Y':
        print('String Distance Matching to LOINC Component and System')

    nrows = unique_combos.shape[0]

    for i in range(nrows):
        if i % 500 == 0 and config.print_status == 'Y':
            print('Matching', i, '/', nrows)
        matches = stringdist.stringdist(unique_combos.at[i, 'TestNameMapJW'], unique_components,
            method='jw', p=0)
        bestmatch = np.argmin(matches)
        unique_combos.at[i, 'PredictedComponentJW'] = unique_components[bestmatch]
        unique_combos.at[i, 'ComponentMatchDistJW'] = matches[bestmatch]

        matches = stringdist.stringdist(unique_combos.at[i, 'TestNameMapLV'], unique_components,
            method='lv')
        bestmatch = np.argmin(matches)
        unique_combos.at[i, 'PredictedComponentLV'] = unique_components[bestmatch]
        unique_combos.at[i, 'ComponentMatchDistLV'] = matches[bestmatch]

        matches = stringdist.stringdist(unique_combos.at[i, 'SpecimenMapJW'], unique_system,
            method='jw', p=0)
        bestmatch = np.argmin(matches)
        unique_combos.at[i, 'PredictedSystemJW'] = unique_system[bestmatch]
        unique_combos.at[i, 'SystemMatchDistJW'] = matches[bestmatch]

        matches = stringdist.stringdist(unique_combos.at[i, 'SpecimenMapLV'], unique_system,
            method='lv')
        bestmatch = np.argmin(matches)
        unique_combos.at[i, 'PredictedSystemLV'] = unique_system[bestmatch]
        unique_combos.at[i, 'SystemMatchDistLV'] = matches[bestmatch]
        
    dat = dat.merge(unique_combos, how='left', left_on=['TestNameMapLV', 'TestNameMapJW', 'SpecimenMapLV',
       'SpecimenMapJW'], right_on=['TestNameMapLV', 'TestNameMapJW', 'SpecimenMapLV',
       'SpecimenMapJW'])
    
    dat.to_csv(config.out_dir + 'datCube.csv', index=False)
    
    return dat

def preproces_source_values(df, col='', source_values=[], value_default=""): 
    # import transformer
    return transformer.preproces_source_values(df=df, col=col, source_values=source_values, value_default=value_default)

def make_string_distance_features(df=None, dataType='test_order_name', loincmap=None, 
       parsed_loinc_fields=None, source_values=[], verbose=1, transformed_vars_only=False, 
       uniq_src_vals=True, value_default="", standardize=False, drop_datatype_col=True, save=True):
    """
 
    Assumptions
    -----------
    1. All input source_values (or df[dataType].values) are of string type, if not, will be converted to strings

    Memo
    ----
    1. Input: T-string value for each row
       Output: The transformed feature values (TestOrderNameMapLV, TestResultNameMapLV, etc)
    """
    def test_match(test_col, test_str, part_name, part_expressions=[], target_terms=[], metric='?'):
        # col: the name of the T-attribute (e.g. test_order_name)
        # term: T-token 
        # part: name of the LOINC part e.g. Component

        if not target_terms: target_terms = ['albumin', 'CD4']

        tTested = False
        max_display = 10
        for target_term in target_terms: 
            if test_str.lower().find(target_term) > 0: 

                print("(test) derived col: {}, term: {} vs part tokens:\n{} ...\n".format(col, test_str, part_expressions[:10]))
                dists = []
                for part_expr in part_expressions: 
                    d = distance_jaro_winkler(test_str, str(part_expr), verbose=0)
                    dists.append(d)

                    if len(dists) < max_display: 
                        print("... T-attribute value: {}:\n... Part expr: {}\n... dist={}\n".format(test_str, part_expr, d))
                    # part_expr: CARDIAC PACEMAKER PROSTHETIC LEAD
                    # test_str: URINE MICROALBUMIN CREATININE RATIO

                bestmatch = np.argmin(dists)
                best_dist = dists[bestmatch]
                best_expr = uniqParts[part_name][bestmatch]
                highlight("(test) min_dist: {} | (col: {}, metric: {}) => best_expr:\n{}\n".format(best_dist, test_col, metric, best_expr))
                
                # [distance_jaro_winkler(term, str(token), verbose=1) for token in part]
                tTested = True
        return tTested

    import stringdist
    from pyjarowinkler import distance
    from loinc import LoincTSet

    # joined_data = build_cube()
    # data = add_cuis_to_cube(joined_data)
    # value_default = "" # LoincTSet.token_default

    if loincmap is None or parsed_loinc_fields is None: 
        # build the loincmap
        loincmap, short_to_long, parsed_loinc_fields = combine_loinc_mapping()
    
    # if df is None: transformed_vars_only = False
    if isinstance(source_values, str): source_values = [source_values, ]
    if len(source_values) > 0: # unique_tests
        transformed_vars_only = True
    else: # only "sourse_values" is given
        assert df is not None and dataType in df.columns, "Neither test{result, order} strings nor training data were given!"
        source_values = df[dataType].values

    # preprocess source value to ensure that all values are of string type
    # preproces_source_values(df, col=dataType, source_values=source_values, value_default=value_default)

    if standardize: source_values = preproces_source_values(source_values=source_values, value_default="")
    ############################################################

    if uniq_src_vals: 
        source_values = np.unique(source_values)
        print("(make_string_distance_features) Found {} unique source values".format(len(source_values)))

    # Find, for each T-string token, the best matched token from the loincmap
    # where T-string refers to the values of {test_order_name, test_result_name, ...}

    # test_match_matrix_JW = load_match_matrix(metric='JW')
    # test_match_matrix_LV = load_match_matrix(metric='LV')

    test_match_matrix_LV, test_match_matrix_JW = get_matches(source_values, loincmap)

    if verbose: 
        # (sdf) string distance feature
        print("(sdf) test_match_matrix_LV (cols={}):\n{}\n".format(test_match_matrix_LV.columns, test_match_matrix_LV.head(50) ))
        print("...   test_match_matrix_JW:\n{}\n".format(test_match_matrix_JW.head(50) ))


    if config.print_status == 'Y': print('Concatenating String Match Results')

    concat_lv_test_match_result = concatenate_match_results(test_match_matrix_LV, dataType, metric='LV')
    concat_jw_test_match_result = concatenate_match_results(test_match_matrix_JW, dataType, metric='JW')

    print("... concat_lv_test_match_result:\n{}\n".format(concat_lv_test_match_result.head(50) ))
    print("... concat_jw_test_match_result:\n{}\n".format(concat_jw_test_match_result.head(50) ))
    #concat_lv_test_match_result.columns.values[0] = LoincTSet.get_sdist_mapped_col_name(dataType, metric="LV")
    #concat_jw_test_match_result.columns.values[0] = LoincTSet.get_sdist_mapped_col_name(dataType, metric="JW")

    df_transformed = DataFrame(source_values, columns=[dataType, ])
    if not transformed_vars_only: assert df is not None, "Source dataframe not given"
        # df_transformed = df

    # col_lv_matched_text = LoincTSet.get_sdist_mapped_col_name(dataType, metric="LV")
    # df_transformed[col_lv_matched_text] = concat_lv_test_match_result[col_lv_matched_text]
    # ... [note] this doesn't work 

    # print("... concat_lv_test_match_result({}):\n{}\n".format(col_lv_matched_text, concat_lv_test_match_result[col_lv_matched_text].values))

    df_transformed = pd.merge(df_transformed, concat_lv_test_match_result, how='left', left_on=dataType, right_index=True)
    # ... join via df_transform[dataType] and index of concat_*

    # col_jw_matched_text = LoincTSet.get_sdist_mapped_col_name(dataType, metric="JW")
    # df_transformed[col_jw_matched_text] = concat_jw_test_match_result[col_jw_matched_text]
    df_transformed = pd.merge(df_transformed, concat_jw_test_match_result, how='left', left_on=dataType, right_index=True)
 
    # concat_test_match_result = pd.concat([concat_lv_test_match_result, concat_jw_test_match_result], axis=1)
    
    print("... token(test) vs token(loinc) | after concatenation:\n{}\n".format(df_transformed.head(100)))
    assert len(source_values) == df_transformed.shape[0]
    df_transformed.fillna("", inplace=True)
    ############################################
    # e.g. 
    #                         test_order_name                   TestOrderMapLV                   TestOrderMapJW
    # 0                                                             NaN => ''                         NaN => ''
    # 1          5 HIAA QUANT 24 HR URINE         5 HIAA QUANT 24 HR URINE         5 HIAA QUANT 24 HR URINE
    # 2               ABO GROUP RH FACTOR              ABO GROUP RH FACTOR              ABO GROUP RH FACTOR
    # 3                  ACCUTYPE R IL28B                ACUTE RIGHT IL28B             ACCUTYPE RIGHT IL28B
    # 4                     ACETAMINOPHEN                    ACETAMINOPHEN                    ACETAMINOPHEN
    
    # dat = data.merge(concat_test_match_result, how='left', left_on='CleanedTestName', right_index=True)
    # dat = dat.merge(concat_spec_match_result, how='left', left_on='CleanedSpecimen', right_index=True)
    # sys.exit(0)
    ############################################
    
    loinc_comp_syst = parsed_loinc_fields[['LOINC', 'Component', 'System']]
    loinc_comp_syst = loinc_comp_syst[(~pd.isnull(loinc_comp_syst.System)) & (loinc_comp_syst.System != '')].reset_index(drop=True)
    # ... reset_index(drop=True): drops the current index of the DataFrame and replaces it with an index of increasing integers

    loinc_comp_syst['ExpandedSystem'] = np.nan
    loinc_comp_syst.ExpandedSystem = loinc_comp_syst.ExpandedSystem.astype(object)
    loinc_num_set = loinc_comp_syst.LOINC.unique()

    if config.print_status == 'Y': print('Generating LOINC System Field Expansion')

    # -- mapping System tokens to full names via loincmap
    rows = loinc_comp_syst.shape[0]
    for i in range(rows):    # foreach row in the loinc table
        if config.print_status == 'Y' and i % 5000 == 0:
            print('Row', i, '/', rows)
        if not pd.isnull(loinc_comp_syst.System[i]):
            loinc_comp_syst.at[i, 'System'] = loinc_comp_syst.System[i].split(" ")
            for term in loinc_comp_syst.System[i]:   # foreach token in System
                mapped_term = loincmap.loc[loincmap.Token == term, 'FinalTokenMap'].values[0]
                # expanded term

                if pd.isnull(loinc_comp_syst.ExpandedSystem[i]):
                    # insert new value
                    loinc_comp_syst.at[i, 'ExpandedSystem'] = mapped_term
                else:
                    loinc_comp_syst.at[i, 'ExpandedSystem'] = loinc_comp_syst.ExpandedSystem[i] + " " + mapped_term

    print("... expanded system:\n{}\n".format(loinc_comp_syst.head(100)))

    cols_sdist_map = LoincTSet.get_sdist_mapped_col_names(dataType, metrics=['LV', 'JW'], throw=True)
    # unique_combos = dat[cols_sdist_map].drop_duplicates().reset_index(drop=True)
    
    uniqParts = {} # partDict
    #######################################################
    uniqParts['Component'] = unique_components = loinc_comp_syst.Component.unique()
    uniqParts['System'] = unique_system = loinc_comp_syst[~pd.isnull(loinc_comp_syst.ExpandedSystem)].ExpandedSystem.unique()
    #######################################################
    print("... unique Component (n={}):\n{}\n".format(len(unique_components), unique_components[:100]))
    print("... unique System (n={}):\n{}\n".format(len(unique_system), unique_system[:100]))

    # ['PredictedComponentJW', 'ComponentMatchDistJW', 
    #  'PredictedComponentLV', 'ComponentMatchDistLV',]
    cols_loinc_parts = LoincTSet.get_sdist_matched_loinc_col_names(dataType, parts=['Component', 'System',], 
           types=['Predicted', 'MatchDist'], metrics=['LV', 'JW'], throw=True)
    print("... derived attributes from input col={}:\n{}\n".format(dataType, cols_loinc_parts))

    # 'TestOrderMapLV', 'TestOrderMapJW' + [ ... ]
    #######################################################
    print("... df_transformed prior to adding predicted, matchdist vars:\n{}\n".format(df_transformed.head(100).to_string(index=False)))
    df_transformed = pd.concat([df_transformed, pd.DataFrame(columns=cols_loinc_parts)], sort=False)
    #######################################################
    
    # numpy2ri.activate()
    # stringdist = importr('stringdist', lib_loc=config.lib_loc)

    if config.print_status == 'Y': print('String Distance Matching to LOINC Component and System')
    
    # --- Test strings to predict LOINC parts (e.g. Component, System)
    #     i.e. features that gauge how well the test strings match with LOINC Component and System

    nrows = df_transformed.shape[0]
    parts = ['Component', 'System', ]
    for i in range(nrows):  # foreach row in the transformed data
        if i % 500 == 0 and config.print_status == 'Y': print('Matching', i, '/', nrows)

        for part in parts: # 'Component', 'System'

            col = LoincTSet.get_sdist_mapped_col_name(dataType, metric="JW")  # mapped test-string in JW
            test_expr = df_transformed.at[i, col] # mapped T-string e.g. "URINE MICROALBUMIN CREATININE RATIO"
            # [check] can be a non-string value or NaN

            assert not pd.isna(test_expr)
            if not isinstance(test_expr, str): 
                test_expr = str(test_expr)
                print("... Found non-string {}-value: {}!".format(col, test_expr))

            #----------------------------------
            # matches = stringdist.stringdist(df_transformed.at[i, 'TestNameMapJW'], unique_components, method='jw', p=0)
            # ... 1.0-distance.get_jaro_distance(test_expr, str(token), winkler=True, scaling=0.1)
            matches = [distance_jaro_winkler(test_expr, str(part_expr), verbose=0) for part_expr in uniqParts[part]]  
            # ... foreach LOINC part expression, compute its distance to test expression
            # ... part_expr: CARDIAC PACEMAKER PROSTHETIC LEAD
            bestmatch = np.argmin(matches)
            #----------------------------------

            # test
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            tTested = test_match(test_col=col, test_str=test_expr, part_name=part, part_expressions=uniqParts[part], metric='JW') 
            # if tTested: sys.exit(0)
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            col_pred = LoincTSet.get_sdist_matched_loinc_col_name(dataType, part=part, vtype='Predicted', metric='JW', throw=True)
            df_transformed.at[i, col_pred] = uniqParts[part][bestmatch]  # uniqParts[part]: {unique_components, unique_systems, ...}

            col_dist = LoincTSet.get_sdist_matched_loinc_col_name(dataType, part=part, vtype='MatchDist', metric='JW', throw=True)
            df_transformed.at[i, col_dist] = matches[bestmatch]

            ###################################################################################################
            # ... JW features ready 

            col = LoincTSet.get_sdist_mapped_col_name(dataType, metric="LV")
            test_expr = df_transformed.at[i, col]  # [check] can be a non-string value???
            if not isinstance(test_expr, str): test_expr = str(test_expr)

            #----------------------------------
            # matches = stringdist.stringdist(df_transformed.at[i, 'TestNameMapLV'], unique_components, method='lv')
            matches = [stringdist.levenshtein(test_expr, str(part_expr)) for part_expr in uniqParts[part]]
            bestmatch = np.argmin(matches)
            #----------------------------------

            col_pred = LoincTSet.get_sdist_matched_loinc_col_name(dataType, part=part, vtype='Predicted', metric='LV', throw=True)
            df_transformed.at[i, col_pred] = uniqParts[part][bestmatch]

            col_dist = LoincTSet.get_sdist_matched_loinc_col_name(dataType, part=part, vtype='MatchDist', metric='LV', throw=True)
            df_transformed.at[i, col_dist] = matches[bestmatch]

            ###################################################
            # ... LV features ready
        
    if save: 
        output_path = os.path.join(config.out_dir, "{}-sdist-vars.csv".format(dataType))
        if verbose: print("(make_string_distance_features) Saving string distance features to:\n{}\n ... #".format(output_path))
        df_transformed.to_csv(output_path, index=False)
        print("(make_string_distance_features) cols(df_transformed):\n{}\n".format(list(df_transformed.columns.values)))

        # e.g. example columns derived from test_order_name
        # ['test_order_name', 'TestOrderMapLV', 'TestOrderMapJW', 'TOPredictedComponentLV', 'TOMatchDistComponentLV', 
        # 'TOPredictedComponentJW', 'TOMatchDistComponentJW', 'TOPredictedSystemLV', 'TOMatchDistSystemLV', 
        # 'TOPredictedSystemJW', 'TOMatchDistSystemJW']

    if transformed_vars_only: 
        return df_transformed

    # otherwise, return the input data (all but the dataType column) plus the attributes derived from the dataType column (e.g. test_result_name)
    print("... before the merge dim(df): {}".format(df.shape))
    df = pd.merge(df, df_transformed, on=dataType)
    print("... after  the merge dim(df): {}".format(df.shape))

    if drop_datatype_col: 
        if verbose: print("(make_string_distance_features) Drop {}, keeping only derived attributes".format(dataType))
        df = df.drop([dataType, ], axis=1)

    return df

def demo_create_distance_vars(save=True): 
    """

    Related
    -------
    mtrt_to_loinc.demo_create_tfidf_vars()
    """
    from analyzer import label_by_performance, col_values_by_codes, load_src_data
    from data_processor import save_data

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
    dfp = dfp.loc[dfp[col_target].isin(target_codes)]

    # unique codes 
    unique_codes = dfp[col_target].unique()
    N_ucodes = len(unique_codes)

    loincmap = load_loincmap(cohort=cohort)
    if loincmap is None: 
        loincmap, short_to_long, parsed_loinc_fields = combine_loinc_mapping()
        # ... byproduct: loincmap-<cohort>.csv

    value_default = ""
    target_test_cols = ['test_order_name', 'test_result_name', ]
    for col in target_test_cols: 
        print("(feature creation) Processing DataType/Column: {} ######\n... dim(data) BEFORE merge: {}\n".format(col, dfp.shape))

        # --- pass df
        # dft = dfp[ [col] ]   # just pass two columns: test_result_loinc_code, test*
        # dft = dft.drop_duplicates().reset_index(drop=True)

        # --- pass only source valus
        dim0 = dfp.shape; N0 = dim0[0]
        dfp = preproces_source_values(df=dfp, col=col, value_default=value_default)
        assert dfp.shape == dim0

        uniq_src_vals = dfp[col].unique()
        N_uniq = len(uniq_src_vals)
        print("... Col: {} => n(unique values): {} | deg(uniq): {} | deg(uniq ~ ncodes): {}".format(col, 
            N_uniq, N_uniq/(N0+0.0), N_uniq/(N_uniq+0.0) ))

        # test_order_names = adict['test_order_name']
        # test_result_names = adict['test_result_name']

        # pass unique test_order_name instead?
        dim0 = dfp.shape
        transformed_vars_only = False # if True, only return the derived features (and the input column but not the rest)
        dfp = make_string_distance_features(df=dfp, 
                    dataType=col,  # this is necessary to construct a dataframe
                    # source_values=uniq_src_vals, # df=dft, dataType='test_order_name', 
                    loincmap=loincmap, # source_values=dfp['test_order_name'].values)
                    drop_datatype_col=True, 
                    transformed_vars_only=transformed_vars_only, 
                    uniq_src_vals=True, value_default=value_default)
        if not transformed_vars_only: 
            assert dfp.shape[1] > dim0[1], "Prior to transformation dim0: {}, after dim: {}".format(dim0, dfp.shape)
        print("... finishing string-matching features | dim(transformed): {} | N0: {}".format(dfp.shape, N0))
        # dft = make_string_distance_features(loincmap=loincmap, source_values=test_order_names)# source_values=dfp['test_order_name'].values)
        
        # merge transformed dataframe with the training data
        # dfp.merge(concat_test_match_result, how='left', left_on='CleanedTestName', right_index=True)
        # assert col in dfp.columns
        # assert col in dft.columns, "col(dft):\n{}\n".format(dft.columns)
        # dfp = pd.merge(dfp, dft, on=col)
        # ... inner join
        print("... dim(data) AFTER merge: {}".format(dfp.shape))
    ### ... 

    # drop the source cols
    # dfp = dfp.drop(target_test_cols, axis=1)
    print("Final dataframe dim: {}, cols: {}".format(dfp.shape, dfp.columns.values))
    
    if save: 
        output_file = f"ts-{cohort}-proc.csv"
        save_data(dfp, output_file=output_file, verbose=1)

    return

def test_csv_to_excel(input_file='', **kargs): 
    """
    Alternatively, just go to Data (tab) -> Text Import to re-format the csv and open it.
    
    Update
    ------
    """
    from pandas.io.excel import ExcelWriter

    # --- example path
    cohort = kargs.get('cohort', 'generic')
    sep = kargs.get('sep', ',')
    verbose = kargs.get('verbose', 1)
    input_dir = kargs.get('input_dir', os.path.join(os.getcwd(), input_dir)) 
    input_file = f"loincmap-{cohort}.csv" 
    input_path = os.path.join(input_dir, input_file)
    
    df.to_csv(input_path, sep=sep, index=False, header=True)

    ext = 'xlsx'
    output_file = input_file.split('.')[0] + ext

    # with ExcelWriter('my_excel.xlsx') as ew:
    #     for csv_file in csv_files:
    pd.read_csv(input_path, sep=sep).to_excel(output_file, sheet_name=input_path)

    return

def test(): 
    # from MapLOINCFields import parse_loinc

    # ### Map LOINC System tokens to LOINC Long Name Tokens
    # short_to_long, parsed_loinc_fields = parse_loinc()
    # system_map = map_loinc_system(parsed_loinc_fields)
    # print("> Map | system token to mapped LN token:\n{}\n".format(tabulate(system_map.head(50), headers='keys', tablefmt='psql')))
    # loinc_terms_max = map_loinc_token_counts(short_to_long)

    target_codes = ['80143', '303503', '273532', '20008', '79095', '244673', '205708','460980',
 '497818', '823765', '19638', '7062', '301804', '142778', '451765', '111252',
 '7138', '51995', '24729', '57828', '24679', '51573', '203943', '81174', '24729',
 '7112', '81232', '728626', '7047', '81166']

    # loincmap = load_loincmap()
    # if loincmap is None: 
    #     loincmap, *rest = combine_loinc_mapping()
    # print(tabulate(loincmap.head(200), headers='keys', tablefmt='psql'))

    # unique_tests = ['LIVER FIBROSIS FIBROMETER', 'AFP TUMOR MARKER', 'CBC W DIFF PLATELET', 'MEASLES MUMPS RUBELLA VARICELLA IGG IGM', ]
    # make_string_distance_features(loincmap=loincmap, source_values=unique_tests)

    # --- features based on string distances
    demo_create_distance_vars()

    return

if __name__ == "__main__": 
    test()



