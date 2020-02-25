import string, sys
import math
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)
 
def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)
 
def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)
 
def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))
 
def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values
 
def tfidf(documents, tokenize):
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents

def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude

def build_tfidf_model(source_values=[], df=None, cols=[], **kargs): 
    from gensim.corpora import Dictionary
    from gensim.models import TfidfModel
    from loinc import LoincTable
    import transformer as tr

    tStandardize = kargs.get('standardize', True)
    value_default = kargs.get('value_default', "")
    ngram_range = kargs.get('ngram_range', (1, 3))
    tVerify = kargs.get('verify', True)
    lowercase = kargs.get('lowercase', False)
    max_features = kargs.get("max_features", 50000)
    # ... if specified, select only the most frequent ordered by term freq

    if len(source_values) == 0:
        assert df is not None 
        if not cols: cols = ['test_result_loinc_code', 'medivo_test_result_type']
        source_values = tr.conjoin(df, cols=cols, transformed_vars_only=True, sep=" ", remove_dup=True)
        # ... remove_dup: if True, remove duplicate tokens in the sentence
        if not tStandardize: 
            print("(build_tfidf_model) Warning: Building corpus from dataframe, you may need to set standardize to True!")

    if tStandardize: 
        source_values = preprocess_text_simple(source_value=source_values, value_default=value_default)

    # source_token_lists = [source_text.split() for source_text in source_values]
    
    # # [test]
    # lengths = [len(source_token_list) for source_token_list in source_token_lists]
    # print("(build_tfidf_model) E[len(texts)]: {}".format( sum(lengths)/(len(lengths)+0.0) ))

    # # -- use gensim
    # dct = Dictionary(source_token_lists)
    # corpus = [dct.doc2bow(tokens) for tokens in source_token_lists]
    # model = TfidfModel(corpus)  # fit model
    # tfidf_matrix =  tfdif.fit_transform([content for file, content in corpus])

    # -- use sklearn
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, min_df=1, 
                stop_words=LoincTable.stop_words, max_features=max_features, lowercase=lowercase) # stop_words/'english'
    tfidf = tfidf.fit(source_values)
    
    # -- test
    if tVerify: 
        fset = tfidf.get_feature_names()
        print("(model) number of features: {}".format(len(fset)))
        Xtr = tfidf.transform(source_values)
        n_display = 30

        analyze = tfidf.build_analyzer()
        np.random.choice(source_values, 1)[0]
        print("(model) ngram_range: {} => {}".format(ngram_range, analyze(np.random.choice(source_values, 1)[0][:100])))

        # --- interpretation 
        print("(model) Interpreting the TF-IDF model")
        tids = set(np.random.choice(range(Xtr.shape[0]), min(Xtr.shape[0], n_display)))
        for i, dvec in enumerate(Xtr):
            if not i in tids: continue
            # top_tfidf_features(dvec, features=tfidf.get_feature_names(), top_n=10)
            df = top_features_in_doc(Xtr, features=fset, row_id=i, top_n=10)
            print("... doc #{}:\n{}\n".format(i, df.to_string(index=True)))

        print("... top n features overall across all docs")
        df = top_mean_features(Xtr, fset, grp_ids=None, min_tfidf=0.1, top_n=10)
        print("... doc(avg):\n{}\n".format(df.to_string(index=True)))

    return tfidf

def find_topn_most_similar(code, model): 
    """

    Memo
    ----
    1. find topn most similar documents: 
       https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity

    2. find similar documents: 
       https://markhneedham.com/blog/2016/07/27/scitkit-learn-tfidf-and-cosine-similarity-for-computer-science-papers/
    """
    return

##########################################################
# --- interpret TF-IDF

def top_features_in_doc(Xtr, features, row_id, top_n=25):
    """
    Top tfidf features in specific document (matrix row)

    Memo
    ----
    1. np.squeeze()

       Remove single-dimensional entries from the shape of an array.
    """
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_features(row, features, top_n)

def top_tfidf_features(row, features, top_n=25):
    """
    Get top n tfidf values in row and return their corresponding feature names.

    Memo
    ----
    1. https://buhrmann.github.io/tfidf-analysis.html
    """
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_mean_features(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    """
    Return the top n features that on average are most important amongst documents in rows
    indentified by indices in grp_ids. 
    """
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_features(tfidf_means, features, top_n)

def top_features_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    """
    Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. 
    """
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_features(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs):
    """
    Plot the data frames returned by the function top_features_by_class(). 
    """
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()

##########################################################

def demo_create_tfidf_vars(save=True):
    from transformer import preprocess_text_simple
    from analyzer import label_by_performance, col_values_by_codes, load_src_data

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
    ############################################################
    # ... now we have the training data with loinc codes of either low classification performance or low sample sizes 

    # loincmap = load_loincmap(cohort=cohort)
    # if loincmap is None: 
    #     loincmap, short_to_long, parsed_loinc_fields = combine_loinc_mapping()
        # ... byproduct: loincmap-<cohort>.csv

    value_default = ""
    target_test_cols = ['test_result_loinc_code', 'medivo_test_result_type', ]
    for col in target_test_cols: 

        # --- pass df
        
        # dft = dfp[ [col] ]   # just pass two columns: test_result_loinc_code, test*
        # dft = dft.drop_duplicates().reset_index(drop=True)

        # --- pass only source valus
        dfp = preprocess_text_simple(dfp, col=col, value_default=value_default)

        
    uniq_src_vals = dfp[col].unique()
    print("... n(unique values): {}".format(len(uniq_src_vals)))

    return

def demo_tfidf_transform(**kargs):
    """

    Memo
    ----

    """
    # from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    # ... compute dot product

    docs = {}
    docs[0] = "SMN1 GENE MUTATION ANALYSIS BLOOD TISSUE MOLECULAR GENETICS METHOD NARRATIVE"
    docs[1] = "SMN1 GENE TARGETED MUTATION ANALYSIS BLOOD TISSUE MOLECULAR GENETICS METHOD"
    docs[2] = "SALMON IGE AB SERUM"
    docs[3] = "SCALLOP IGE AB RAST CLASS SERUM"
    docs[4] = "SJOGRENS SYNDROME A EXTRACTABLE NUCLEAR AB SERUM"
    docs[5] = "MYELOCYTES BLOOD"

    dtest = {}
    dtest[0] = "SCALLOP IGE AB RAST CLASS SERUM"

    corpus = np.array([docs[i] for i in range(len(docs))])
    vectorizer = CountVectorizer(decode_error="replace")
    vec_train = vectorizer.fit_transform(corpus)

    # -- model persistance
    # # Save vectorizer.vocabulary_
    # pickle.dump(vectorizer.vocabulary_,open("feature.pkl","wb"))

    # # Load it later
    # transformer = TfidfTransformer()
    # loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
    # tfidf = transformer.transform(loaded_vec.fit_transform(np.array(["aaa ccc eee"])))

    # vec = TfidfVectorizer()
    # tfidf = vec.fit_transform()

    ngram_range = (1,3)
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, min_df=0, smooth_idf=True) 
    # sublinear_tf=True? it's unlikely to observe repeated tokens in the LOINC long name or MTRT

    Xtr = tfidf.fit_transform(corpus)
    
    analyze = tfidf.build_analyzer()
    print("... ngram_range: {} => {}".format(ngram_range, analyze("RHEUMATOID FACTOR IGA SERUM")))

    # --- get feature index
    part_sent = "CLASS SERUM"
    feature_index = tfidf.vocabulary_.get("CLASS SERUM".lower())  # lowercase: True by default
    print("... phrase: {} => {}".format(part_sent, feature_index))

    # > size of the vocab
    # tfidf.vocabulary_: a dictionary
    print("... size(vocab): {}".format( len(tfidf.vocabulary_) ))

    # -- doc vectors
    # print("... d2v(train):\n{}\n".format( tfidf.to_array() ))
    fset = tfidf.get_feature_names()
    print("> feature names:\n{}\n".format(fset))
    for i, dvec in enumerate(Xtr):
        print("> doc #[{}]:\n{}\n".format(i, dvec.toarray()))


    # --- predicting new data
    corpus_test = np.array([doc for i, doc in dtest.items()])
    doc_vec_test = tfidf.transform(corpus_test)
    print("... d2v(test):\n{}\n".format( doc_vec_test.toarray() ))

    # --- interpretation 
    print("(demo_predict) Interpreting the TF-IDF model")
    for i, dvec in enumerate(Xtr):
        # top_tfidf_features(dvec, features=tfidf.get_feature_names(), top_n=10)
        df = top_features_in_doc(Xtr, features=fset, row_id=i, top_n=10)
        print("... doc #{}:\n{}\n".format(i, df.to_string(index=True)))

    print("... top n features overall across all docs")
    df = top_mean_features(Xtr, fset, grp_ids=None, min_tfidf=0.1, top_n=10)
    print("... doc(avg):\n{}\n".format(df.to_string(index=True)))

    # --- interface
    # a. get the scores of individual tokens or n-grams in a given document? 
    df = pd.DataFrame(Xtr.toarray(), columns = tfidf.get_feature_names())
    vocab = ['salmon ige ab', 'salmon']


    return

def demo_tfidf(**kargs):
    """


    Memo
    ----
    TfidfVectorizer is equivalent to CountVectorizer followed by TfidfTransformer, where

    CountVectorizer: Transforms text into a sparse matrix of n-gram counts.
    TfidfTransformer: Performs the TF-IDF transformation from a provided matrix of counts.
    """
    # import string, sys
    # import math
    # from sklearn.feature_extraction.text import TfidfVectorizer

    tokenize = lambda doc: doc.upper().split(" ")
 
    document_0 = "SMN1 GENE MUTATION ANALYSIS BLOOD TISSUE MOLECULAR GENETICS METHOD NARRATIVE"
    document_1 = "SMN1 GENE TARGETED MUTATION ANALYSIS BLOOD TISSUE MOLECULAR GENETICS METHOD"
    document_2 = "SALMON IGE AB SERUM"
    document_3 = "SCALLOP IGE AB RAST CLASS SERUM"
    document_4 = "SJOGRENS SYNDROME A EXTRACTABLE NUCLEAR AB SERUM"
    document_5 = "MYELOCYTES BLOOD"
    document_6 = "KAPPA LIGHT CHAINS FREE 24 HOUR URINE"
 
    all_documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]
     
    sklearn_tfidf = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
    # sublinear_tf: Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf)
    # smooth_idf: Smooth idf weights by adding one to document frequencies, as if an extra document 
    #             was seen containing every term in the collection exactly once. Prevents zero divisions.

    tfidf_representation = tfidf(all_documents, tokenize)
    sklearn_representation = sklearn_tfidf.fit_transform(all_documents) 
    # sklearn_representation: a sparse matrix

    # print(tfidf_representation[0])
    # print(sklearn_representation.toarray()[0].tolist())

    our_tfidf_comparisons = []
    for count_0, doc_0 in enumerate(tfidf_representation):
        for count_1, doc_1 in enumerate(tfidf_representation):
            our_tfidf_comparisons.append((cosine_similarity(doc_0, doc_1), count_0, count_1))

    skl_tfidf_comparisons = []
    for count_0, doc_0 in enumerate(sklearn_representation.toarray()):
        for count_1, doc_1 in enumerate(sklearn_representation.toarray()):
            skl_tfidf_comparisons.append((cosine_similarity(doc_0, doc_1), count_0, count_1))

    for x in zip(sorted(our_tfidf_comparisons, reverse = True), sorted(skl_tfidf_comparisons, reverse = True)):
        print(x)

    return

def demo_text_model(): 
    from feature_gen_sdist import distance_jaro_winkler
   
    # string matching-based distance metrics
    cases = [("", ""), ("CBC W DIFF PLATELET COUNT", ""), ("CBC W DIFF PLATELET COUNT", "CBC PLATELET COUNT"), 
             ("CBC W DIFF PLATELET COUNT", "CBC W DIFF PLATELET"), ("CBC W DIFF", "CBC W DIFF PLATELET COUNT")
             ]
    for i, (x, y) in enumerate(cases): 
        d = distance_jaro_winkler(x, y, verbose=1)
        print("> JW distance |\nx={}\ny={}\nd={}".format(x, y, d))

    return

def test(): 

    # --- Text features in general 
    # demo_text_model()

    # --- TF-IDF encoding
    # demo_tfidf()

    # --- prediction using the vectors produced by TF-IDF encoding 
    demo_tfidf_transform()

    return

if __name__ == "__main__": 
    test()