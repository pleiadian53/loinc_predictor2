# coding: utf-8


import re, os
import config
import pandas as pd
from collections import defaultdict, Counter

REJECTION_THRESHOLD = config.rejection_threshold

def import_source_data(verbose=1, warn_bad_lines=False, sep=',', filter_site=True): 
    from loinc import LoincTSet 

    if config.print_status == 'Y':
        print('Importing source data')

    testResultNameList = defaultdict(list)  # testNameList
    testOrderNameList = defaultdict(list)
    specimenList = defaultdict(list)

    site = config.site
    test_order_name = config.test_order_col
    test_result_name = config.test_result_col
    test_specimen_type = config.spec_col
    dtypes = config.dtypes

    target_sites = set(LoincTSet.meta_sender_name)

    df = pd.read_csv(config.in_file, sep=sep, header=0, index_col=None, 
               error_bad_lines=False, warn_bad_lines=warn_bad_lines, dtype=dtypes)
    print("(import_source_data) columns:\n{}\n".format( list(df.columns.values) ))

    skipped_sites = set([])
    for r, row in df.iterrows(): 
        if r > 0 and (r % 50000 == 0): print("##### Processing line #{} ...".format(r))

        if filter_site and (row[site] not in target_sites): 
            # print("... {} not in {}".format(row[site], target_sites))
            skipped_sites.add(row[site])
            continue

        # test_result_name
        value = str(row[test_result_name]).upper().strip()
        if ((row[site] not in testResultNameList) or 
               (row[site] in testResultNameList and value not in testResultNameList[row[site]])):
            testResultNameList[row[site]].append(value)

        # test_order_name
        value = str(row[test_order_name]).upper().strip()
        if ((row[site] not in testOrderNameList) or
                (row[site] in testOrderNameList and value not in testOrderNameList[row[site]])):
            testOrderNameList[row[site]].append(value)

        value = str(row[test_specimen_type]).upper().strip()
        if ((row[site] not in specimenList) or 
                (row[site] in specimenList and value not in specimenList[row[site]])):
            specimenList[row[site]].append(value)

    cleanedOrders = clean_terms(testOrderNameList, 'test_order_name')
    cleanedResults = clean_terms(testResultNameList, 'test_result_name') # testResuLtnameList: site -> test names
    cleanedSpecimens = clean_terms(specimenList, 'test_specimen_type')

    print("(import_source_data) Skipped {} sites/senders:\n{}\n".format(len(skipped_sites), skipped_sites))
        
    return [cleanedOrders, cleanedResults, cleanedSpecimens]

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

def preprocess_terms(df, dataType='testNames'):
    """
    
    Memo
    ----
    1. Some test names look like LOINC codes, CPT codes etc.
    """
    pass 

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
            modTerm = (term.replace("'", "").replace(",", " ").replace(".", " ") \
                .replace(":", " ").replace('\t', " ").replace("^", " ").replace("+", " ")\
                .replace("*", " ").replace("~", " ").replace("(", " ").replace(")", " ")\
                .replace("!",  " ").replace("[", " ").replace("]", " ")\
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
        
    if dataType in ["test_order_name", "test_result_name", ]:   # testNames
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
    if dataType == "test_result_name":
        cols = ['Site', 'OriginalTestResult', 'CleanedTestResult']
    elif dataType == "test_order_name":
        cols = ["Site", "OriginalTestOrder", "CleanedTestOrder"]
    else:
        cols = ['Site', 'OriginalSpecimen', 'CleanedSpecimen']
    # cols = ['site', 'original', 'cleaned', ]
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

