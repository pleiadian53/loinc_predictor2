from MapLOINCFields import clean_terms, expand_words, add_match


def parse_test_tokens_with_loinc(df, col_target='test_result_loinc_code', **kargs): # test_order_name, test_result_name
    import loinc as lc

    # andromeda-pond-hepatitis-c-balanced.csv has illed-formed data
    cohort = kargs.get('cohort', 'hepatitis-c')
    token_default = 'unknown'
    df = load_data(input_file='andromeda-pond-{cohort}.csv', warn_bad_lines=False, canonicalized=True)

    # find the cases where LOINCs are given and either test_result_name, test_order_name have values
    


    reader = csv.reader(open(loincFilePath, encoding='utf8'))
    index = -1
    loincs = list()
    shortToLong = defaultdict(Counter)
    componentParsed = list()
    systemParsed = list()
    longParsed = list()
    for fields in reader:
        index = index + 1
        if index == 0:
            loincNumInd = fields.index('LOINC_NUM')
            componentInd = fields.index('COMPONENT')
            systemInd = fields.index('SYSTEM')
            shortNameInd = fields.index('SHORTNAME')
            longNameInd = fields.index('LONG_COMMON_NAME')
            classTypeInd = fields.index('CLASSTYPE')
            continue

        loincNum = fields[loincNumInd]
        component = fields[componentInd].upper()
        system = fields[systemInd].upper();
        shortName = fields[shortNameInd].upper();
        longName = fields[longNameInd].upper();
        classType = fields[classTypeInd];

        if classType != "1" and classType != "2": continue  #only keep the lab and clinical class types

        loincs.append(loincNum)
        shortWords = clean_terms(shortName)
        
        longName = re.sub(r"\[([A-Za-z0-9]*\s*\/*)*\]", "", longName)  # remove brackets
        
        longWords = clean_terms(longName)
        componentWords = clean_terms(component)
        systemWords = clean_terms(system)

        componentParsed.append(" ".join(componentWords))
        systemParsed.append(" ".join(systemWords))
        longParsed.append(" ".join(longWords))

        shortToLong = expand_words(shortToLong, shortWords, longWords) # expand short name to long name
    ### end foreach record in LOINC table

    short_to_long_df = pd.DataFrame(data=[[outer_key, inner_key, shortToLong[outer_key][inner_key]] 
                                            for outer_key in shortToLong for inner_key in shortToLong[outer_key]], 
                        columns=['Token', 'TokenMap', 'Count'])
    # parsed_loinc_fields_df = pd.DataFrame(data=list(zip(loincs, componentParsed, systemParsed, longParsed)),
    #     columns=['LOINC', 'Component', 'System', 'LongName'], dtype=object)

    if canonicalized: 
        parsed_loinc_fields_df = lc.canonicalize(parsed_loinc_fields_df, col_target="LOINC", verbose=1)

    if config.write_file_loinc_parsed:
        short_to_long_df.to_csv( os.path.join(config.out_dir, "LOINC_Name_Map.csv"), sep="|", index=False)
        parsed_loinc_fields_df.to_csv(  os.path.join(config.out_dir, "LOINC_Parsed_Component_System_Longword.csv"), sep="|", index=False)
    return [short_to_long_df, parsed_loinc_fields_df]

