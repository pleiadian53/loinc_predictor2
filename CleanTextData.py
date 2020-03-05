# coding: utf-8


import re, os
import config
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

from loinc import LoincTSet, LoincTable

REJECTION_THRESHOLD = config.rejection_threshold

def standardize(x): 
    if pd.isna(x): 
        return ""
    if not isinstance(x, str): 
        return str(x)
    return x.upper().strip()

def resolve_test_value(row, x, value_col='test_result_value'): 
    if pd.isna(x): 
        # noop
        return ''
    if isinstance(x, (float, int)): # if the value from test_result_name is a number
        if isinstance(row[value_col], (float, int)):
            if row[value_col] != x:
                print("(value) Found inconsistent lab values | col({}): {} <> {}".format(col, row[col], x))
                return str(x)  # then x is probably not meant for a test result value
            else: 
                return ''   # test_result_value already has the info
        elif isinstance(row[value_col], str): # then this probably should have been in test_result_name 
            print("(value) Found text in {} => write it back to {}".format(value_col, name_col))
            return row[value_col]  # test_result_value should have filled in test_result_name instead
    return x

def import_source_data(input_path='', verbose=1, warn_bad_lines=False, sep=',', filter_site=False, save=True, backup=True): 
    """
    
 
    """
    from loinc import LoincTSet 

    if config.print_status == 'Y':
        print('Importing source data')

    testResultNameList = defaultdict(list)  # testNameList
    testOrderNameList = defaultdict(list)
    specimenList = defaultdict(list)
    commentList = defaultdict(list)
    mtrtList = defaultdict(list)

    # testResultNameIndex = defaultdict(list) 
    # testOrderNameIndex = defaultdict(list)
    # specimenIndex = defaultdict(list)
    # commentIndex = defaultdict(list)    
    
    site = config.site
    test_order_name = config.test_order_col
    test_result_name = config.test_result_col
    test_specimen_type = config.spec_col
    test_result_comments = config.test_comment_col
    medivo_test_result_type = config.tagged_col

    dtypes = config.dtypes
    target_sites = set(LoincTSet.meta_sender_names)

    if not input_path: input_path = config.in_file
    df = pd.read_csv(input_path, sep=sep, header=0, index_col=None, error_bad_lines=False, warn_bad_lines=warn_bad_lines, dtype=dtypes)
    print("(import_source_data) columns:\n{}\n".format( list(df.columns.values) ))

    skipped_sites = set([])
    skipped_index = []

    cached = defaultdict(list) # store cleaned text row by row

    for r, row in df.iterrows(): 
        if r > 0 and (r % 50000 == 0): print("##### Processing line #{} ...".format(r))

        if filter_site and (row[site] not in target_sites): 
            # print("... {} not in {}".format(row[site], target_sites))
            skipped_sites.add(row[site])
            skipped_index.append(r)
            # test_order_name_cleaned(row[])
            continue

        # test_order_name
        value = standardize(row[test_order_name]) # str(row[test_order_name]).upper().strip()
        cached[test_order_name].append(clean_term(value))
        # ... note that collecting invidivdual cleaned value should not be necessary but somehow update_values() is not perfect
        if ((row[site] not in testOrderNameList) or
                (row[site] in testOrderNameList and value not in testOrderNameList[row[site]])):
            testOrderNameList[row[site]].append(value)  # standardize(row[test_order_name])

        # test_result_name
        value = standardize(row[test_result_name]) # str(row[test_result_name]).upper().strip()
        value = resolve_test_value(row, value)
        cached[test_result_name].append(clean_term(value))
        if ((row[site] not in testResultNameList) or 
               (row[site] in testResultNameList and value not in testResultNameList[row[site]])):
            testResultNameList[row[site]].append(value)  # standardize(row[test_result_name])

        # test_result_comments 
        value = standardize(row[test_result_comments]) # str(row[test_result_comments]).upper().strip()
        value = resolve_test_value(row, value)
        cached[test_result_comments].append(clean_term(value))  # test_result_comments_cleaned.append(clean_term(value))
        if ((row[site] not in commentList) or
                (row[site] in commentList and value not in commentList[row[site]])):
            commentList[row[site]].append(value) # standardize(row[test_result_comments]

        # test_specimen_type
        value = standardize(row[test_specimen_type]) # str(row[test_specimen_type]).upper().strip()
        cached[test_specimen_type].append(clean_term(value))
        if ((row[site] not in specimenList) or 
                (row[site] in specimenList and value not in specimenList[row[site]])):
            specimenList[row[site]].append(value)  # standardize(row[test_specimen_type])

        # medivo_test_result_type (MTRT)
        value = standardize(row[medivo_test_result_type]) # str(row[medivo_test_result_type]).upper().strip()
        cached[medivo_test_result_type].append(clean_term(value))
        if ((row[site] not in mtrtList) or 
                (row[site] in mtrtList and value not in mtrtList[row[site]])):
            mtrtList[row[site]].append(value)  # standardize(row[medivo_test_result_type])

    mapped = {}
    mapped['test_order_name'] = cleanedOrders = clean_terms(testOrderNameList, 'test_order_name')
    mapped['test_result_name'] = cleanedResults = clean_terms(testResultNameList, 'test_result_name') # testResuLtnameList: site -> test names
    mapped['test_specimen_type'] = cleanedSpecimens = clean_terms(specimenList, 'test_specimen_type')
    mapped['test_result_comments'] = cleanedComments = clean_terms(commentList, 'test_result_comments')
    mapped['medivo_test_result_type'] = cleanedMTRT = clean_terms(mtrtList, 'medivo_test_result_type')

    if save: 
        # if backup:
        #     if verbose: print("(import_source_data) Saving a backup copy of the raw input to:\n{}\n".format(config.backup_file))
        #     df.to_csv(config.backup_file, sep=sep, index=False, header=True)
        # if verbose: print("(import_source_data) Skipped {} sites/senders:\n{}\n".format(len(skipped_sites), skipped_sites))
        
        output_path = config.processed_file
        if verbose: print("(import_source_data) Saving new training data (with cleaned values) at:\n{}\n".format(output_path))        
        df = update_values(df, cached=cached)
        df.to_csv(output_path, sep=sep, index=False, header=True)

    return mapped

def import_source_data0(verbose=1):
    if config.print_status == 'Y':
        print('Importing source data')
    testResultNameList = defaultdict(list)  # testNameList
    testOrderNameList = defaultdict(list)
    specimenList = defaultdict(list)
    
    reader = open(config.in_file, 'r')
    print("(import_source_data) Reading source data from:\n{}\n".format(config.in_file))
    index = -1
    for i, line in enumerate(reader):
        if i > 0 and (i % 50000 == 0): print("##### Processing line #{} ...".format(i))
        fields = line.split(config.delim)
        if index == -1:
            siteIdentifierCol = fields.index(config.site)  # use meta_sender_name as site name
            testOrderCol = fields.index(config.test_order_col)
            testResultCol = fields.index(config.test_result_col)
            specimenCol = fields.index(config.spec_col)
        index = index + 1
        if index == 0: continue

        # testResultNameList (multibag): site (meta sender name) -> [ test result names ]

        if ((fields[siteIdentifierCol] not in testResultNameList) or 
               (fields[siteIdentifierCol] in testResultNameList and
                    fields[testResultCol].upper().strip() not in testResultNameList[fields[siteIdentifierCol]])):
            testResultNameList[fields[siteIdentifierCol]].append(fields[testResultCol].upper().strip())

        if ((fields[siteIdentifierCol] not in testOrderNameList) or
                (fields[siteIdentifierCol] in testOrderNameList and
                    fields[testOrderCol].upper().strip() not in testOrderNameList[fields[siteIdentifierCol]])):
            testOrderNameList[fields[siteIdentifierCol]].append(fields[testOrderCol].upper().strip())

        if ((fields[siteIdentifierCol] not in specimenList) or 
                (fields[siteIdentifierCol] in specimenList and
                    fields[specimenCol].upper().strip() not in specimenList[fields[siteIdentifierCol]])):
            specimenList[fields[siteIdentifierCol]].append(fields[specimenCol].upper().strip())

    cleanedOrders = clean_terms(testOrderNameList, 'test_order_name')
    cleanedResults = clean_terms(testResultNameList, 'test_result_name') # testResuLtnameList: site -> test names
    cleanedSpecimens = clean_terms(specimenList, 'test_specimen_type')

    return [cleanedOrders, cleanedResults, cleanedSpecimens]
    # return {'test_result_order': cleanOrders, 'test_result_name': cleanedResults, ''}

def update_values(df, dtypes=[], sep='|', cached={}):
    import pandas as pd
    from loinc import LoincTSet 

    if len(cached) > 0: 
        for col, values in cached.items(): 
            df[col] = values
        return df
    
    # --- otherwise, read the cleaned values from the file

    if not dtypes: dtypes = ['test_order_name', 'test_result_name', 'test_specimen_type', 'test_result_comments', 'medivo_test_result_type', ]
    for dtype in dtypes: 
        print("(update_values) Processing dtype: {} ######".format(dtype))
        fpath = os.path.join(config.out_dir, "cleaned_{}.csv".format(dtype))
        df_cleaned = pd.read_csv(fpath, sep=sep, header=0, index_col=None, error_bad_lines=True)
        n_cleaned = df_cleaned.shape[0]
        cols_cleaned = LoincTSet.get_cols_cleaned(dtype)

        test_cases = np.random.choice( range(df_cleaned.shape[0]), min(n_cleaned, 100))
        for r, row in df_cleaned.iterrows(): 
            orig, cleaned = row[cols_cleaned[1]], row[cols_cleaned[2]]
            if not pd.isna(orig) and len(str(orig)) > 0: 
                if r in test_cases: 
                    df_matched = df.loc[df[dtype].apply(standardize) == orig]
                    assert not df_matched.empty, "Debug: Wrong columns? {} | sample values:\n{}\n... orig: {} | cleaned: {}".format(
                            cols_cleaned, df.sample(n=min(10, df.shape[0]))[dtype], orig, cleaned)
                    print("... Found {} matches | orig: {} | cleaned: {}".format(df_matched.shape[0], orig, cleaned))
                # [todo] some of the text could not find a match, why?

                # df.loc[df[dtype].str.upper().str.strip() == orig, dtype] = cleaned
                df.loc[df[dtype].apply(standardize) == orig, dtype] = cleaned

                # df.loc[df[dtype] == orig, dtype] = cleaned   # replace with cleaned value
    return df

def preprocess_terms(df, dataType='testNames'):
    """
    
    Memo
    ----
    1. Some test names look like LOINC codes, CPT codes etc.
    """
    pass 

def clean_term(term, site='', siteWordCount=None, dataType=''): # dtype
    """

    Memo
    ---- 
    1. siteWordCount is expected to be given externally

        siteWordCount = defaultdict(Counter)
        siteTotalWordCount = defaultdict(int)

    """
    if pd.isna(term): 
        print("(clean_term) Input term is NaN: {}".format(term))
        return ''
    if not isinstance(term, str): 
        return str(term)

    insigWords = LoincTable.stop_words # ["IN", "FROM", "ON", "OR", "OF", "BY", "AND", "&", "TO", "BY", "", " "]
    
    modTerm = (term.replace("'", "").replace(",", " ").replace(".", " ") \
        .replace(":", " ").replace('\t', " ").replace("^", " ").replace("+", " ")\
        .replace("*", " ").replace("~", " ").replace("(", " ").replace(")", " ")\
        .replace("!",  " ").replace("[", " ").replace("]", " ").replace("{", " ").replace("}", " ")\
        .replace("_", " ").replace("|", " ").replace('"', " ").split(" "))

    #############################################################################
    i = 0
    while i < len(modTerm):
        modTerm[i] = re.sub(r"\d{1,2}[\/-]\d{1,4}([\/-]\d{2,4})*|\d{6}", "", modTerm[i])
        if modTerm[i] != None and len(modTerm[i]) > 0:
            i = i + 1
        else:
            modTerm.remove(modTerm[i])
    #############################################################################

    # remove repeated tokens 
    modTerm = sorted(set(modTerm), key=modTerm.index)

    j = 0
    nameSplit = list()
    while j < len(modTerm):
        splits = modTerm[j].replace("/", " ").replace("\\", " ").replace("-", " ").split(" ")
        k = 0
        while ((k < len(splits)) and (len(splits[k]) > 0) and (splits[k] not in insigWords)):
            newWord = splits[k].strip()
            nameSplit.append(newWord)

            if len(site) > 0 and isinstance(siteWordCount, dict): 
                siteWordCount[site][newWord] += 1
            k = k + 1
        j = j + 1

    return " ".join(nameSplit)

def clean_terms(sourceData, dataType):
    if config.print_status == 'Y':
        print('Cleaning source data for data type: {}'.format(dataType))
    insigWords = ["IN", "FROM", "ON", "OR", "OF", "BY", "AND", "&", "TO", "BY", "", " "]
    siteWordCount = defaultdict(Counter)
    siteTotalWordCount = defaultdict(int)
    cleanedList = defaultdict(lambda: defaultdict(list))
    discardedTerms = defaultdict(list)
    for siteKey in sourceData.keys():
        for term in sourceData[siteKey]:
            if pd.isna(term): 
                print("(clean_terms) Input term is NaN: {}".format(term))
                continue 
            if not isinstance(term, str): 
                print("(clean_terms) Input term is not a string: {}".format(term))
                continue

            modTerm = (term.replace("'", "").replace(",", " ").replace(".", " ") \
                .replace(":", " ").replace('\t', " ").replace("^", " ").replace("+", " ")\
                .replace("*", " ").replace("~", " ").replace("(", " ").replace(")", " ")\
                .replace("!",  " ").replace("[", " ").replace("]", " ").replace("{", " ").replace("}", " ")\
                .replace("_", " ").replace("|", " ").replace('"', " ").split(" "))

            #############################################################################
            i = 0
            while i < len(modTerm):
                modTerm[i] = re.sub(r"\d{1,2}[\/-]\d{1,4}([\/-]\d{2,4})*|\d{6}", "", modTerm[i])
                if modTerm[i] != None and len(modTerm[i]) > 0:
                    i = i + 1
                else:
                    modTerm.remove(modTerm[i])
            #############################################################################

            # remove repeated tokens 
            modTerm = sorted(set(modTerm), key=modTerm.index)

            j = 0
            nameSplit = list()
            while j < len(modTerm):
                splits = modTerm[j].replace("/", " ").replace("\\", " ").replace("-", " ").split(" ")
                k = 0
                while ((k < len(splits)) and (len(splits[k]) > 0) and (splits[k] not in insigWords)):
                    newWord = splits[k].strip()
                    nameSplit.append(newWord)
                    siteWordCount[siteKey][newWord] += 1
                    k = k + 1
                j = j + 1

            if siteKey not in cleanedList.keys():
                cleanedList[siteKey][term] = nameSplit
            if term not in cleanedList[siteKey].keys():
                cleanedList[siteKey][term] = nameSplit
    
    for site in siteWordCount.keys():
        siteTotalWordCount[site] = sum(siteWordCount[site].values())
        
    if dataType in ["test_order_name", "test_result_name", 'test_result_comments', 'medivo_test_result_type', ]:   # testNames
        if REJECTION_THRESHOLD is not None:
            filter_out_frequent_tokens(cleanedList, siteWordCount, siteTotalWordCount, discardedTerms)
        cleanedList = convert_to_df(cleanedList, dataType)
        if config.write_file_source_data_cleaning:
            # cleanedList.to_csv( os.path.join(config.out_dir, "Cleaned_Lab_Names.csv"), sep='|', index=False)
            cleanedList.to_csv( os.path.join(config.out_dir, "cleaned_{}.csv".format(dataType)), sep='|', index=False)
            # write_word_ct_csv( os.path.join(config.out_dir, "_Site_Lab_Word_Count.csv"), siteWordCount)
            write_word_ct_csv( os.path.join(config.out_dir, "by_site_word_count_{}.csv".format(dataType)), siteWordCount)
            if len(discardedTerms) > 0:
                # write_discarded_terms( os.path.join(config.out_dir, "Discarded_Lab_Names.csv"), discardedTerms)
                write_discarded_terms( os.path.join(config.out_dir, "discarded_{}.csv".format(dataType)), discardedTerms)

    if dataType in ["test_specimen_type", ]:
        cleanedList = convert_to_df(cleanedList, dataType)
        if config.write_file_source_data_cleaning:
            # cleanedList.to_csv( os.path.join(config.out_dir, "Cleaned_Specimen_Names.csv"), sep='|', index=False)
            cleanedList.to_csv( os.path.join(config.out_dir, "cleaned_{}.csv".format(dataType)), sep='|', index=False)
            # write_word_ct_csv( os.path.join(config.out_dir, "By_Site_Specimen_Word_Count.csv"), siteWordCount)
            write_word_ct_csv( os.path.join(config.out_dir, "by_site_word_count_{}.csv".format(dataType)), siteWordCount)

    return cleanedList

def convert_to_df(cleanedList, dataType):
    # cols = ['site', 'original', 'cleaned', ]
    cols = LoincTSet.get_cols_cleaned(dataType)

    # outer_key: Site
    return pd.DataFrame(([outer_key, inner_key, 
            " ".join(cleanedList[outer_key][inner_key])] for outer_key in cleanedList.keys() 
                for inner_key in cleanedList[outer_key].keys()),
                    columns=cols)

def filter_out_frequent_tokens(cleanedTestNameList, siteWordCount, siteTotalWordCount, discardedTerms):
    for site in siteWordCount.keys():
        for token in siteWordCount[site].keys():
            siteWordCtPct = 100.0 * siteWordCount[site][token] / siteTotalWordCount[site]
            if (siteWordCtPct > REJECTION_THRESHOLD) and (token != "%") and (token != "#"):
                for key in cleanedTestNameList[site].keys():
                    if token in cleanedTestNameList[site][key]:
                        cleanedTestNameList[site][key].remove(token)
                if ((site not in discardedTerms.keys()) or (token not in discardedTerms[site])):
                    discardedTerms[site].append(token)

def write_cleaned_terms(pathName, data):
    data.to_csv(pathName, sep='|')

def write_word_ct_csv(pathName, data):
    with open(pathName, 'w') as out_file:
        out_file.write("Site|Term|Count|Percent\n")
        for site in data.keys():
            total_num = sum(data[site].values())
            for word in data[site].keys():
                count = data[site][word]
                percent = 100.0 * data[site][word] / total_num
                out_file.write(site + "|" + word + "|" + str(count) + "|" + str(percent) + "\n")

def write_discarded_terms(pathName, discardedTerms):
    with open(pathName, 'w') as out_file:
        out_file.write("Site|DiscardedName\n")
        for site in discardedTerms.keys():
            for term in discardedTerms[site]:
                out_file.write(site + "|" + term + "\n")

def test(**kargs):

    import_source_data()

    return  

if __name__ == "__main__": 
    test()

