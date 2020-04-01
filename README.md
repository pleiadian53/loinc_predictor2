Introduction and extended documentation
---------------------------------------

Standards and accuracy for the Logical Observation Identifiers Names and Codes (LOINC) are critical for interoperability and data sharing. In addition, many disease phenotyping analytics are also contingent upon the accuracy of the LOINC codes. However, ... (please more at 
https://prognos.atlassian.net/wiki/spaces/DS/pages/466649366/Loinc+Predictor)


Clone and reproduce results
---------------------------

````
$ git clone https://github.com/medivo/loinc_predictor2
$ cd loinc_predictor2
$ conda env create -f environment.yml
````

Prerequisite
------------
For ease of illustration, we shall assume that the package (loinc_predictor) is installed under 
`<project_dir>/loinc_predictor`, where project_dir is the directory of your choice hosting this module.


**1. Non-standard modules**: 

Please install the following dependent packages:


1a. General purposes: 
   
   - [tabulate](https://pypi.org/project/tabulate/): pretty prints tabular data such as pandas dataframe

1b. For visuailzing decision tree:

   - [pydotplus](https://pydotplus.readthedocs.io/): provides a Python Interface to Graphviz’s Dot language
   - [graphviz](https://www.graphviz.org/): an open-source graph visualization software
        
      e.g. conda install graphviz
   
1c. String matching algorithms: 

   - [fuzzywuzzy](https://pypi.org/project/fuzzywuzzy/): computes distance between two sequences based on Levenshtein Distance
   - [pyjarowinkler](https://pypi.org/project/pyjarowinkler/): computes Jaro-Winkler similarity
   - [StringDist](https://pypi.org/project/StringDist/): computes Levenshtein distance & restricted Damerau-Levenshtein distance

   Optional packages: 

   - [feature_selector](https://pypi.org/project/feature-selector/): Used to identify and select important features for ML algorithms
   - [gensim](https://pypi.org/project/gensim/): A library for information retrival, topic modeling, string comparisons and other NLP tasks 
   

**2. Input data**:

Training LOINC predictive models requires input training data. These data are assumed to 
have been made available and kept under `<project_dir>/loinc_predictor/data`

An example dataset sampled from Andromeda specific to the Hepatitis-C cohort is included under:

        data/andromeda-pond-hepatitis-c.csv.fake 

Note that due to the limit of file size and the sensitivity of the data, we are unable to host physical copies of certains file directly but instead, a link to physical file on the Amazon S3 bucket is included within these files. All the data files suffixed by .fake are such files including the example training data mentioned above.

Data can be curated from subsampling Andromeda datalake. A few tips on the training data curation. 

It is highly recommended that the training data be prepared based on the disease of interest since there are large and ever-growing 
number of LOINC codes as more laboratoy tests and standards are introduced to the healthcare system. For instannce, a Hepatitis-C cohort 
can be linked to at least 700 or more LOINC codes, making it challenging for predictive analytics due to the high cardinality of class
labels (i.e. multiclass classfication problem with large number of classes). Therefore, we shall train the LOINC predictor on a disease-specific 
basis. 

Once the target disease is given, we are now ready to gather data from Andromeda datalake. A Hepatitis-C dataset would comprise sampled rows of patient data from Andromeda that match a set of ICD codes pertaining to Hepatitis C. Please refer to [Clinical Classfication Software](https://www.hcup-us.ahrq.gov/tools_software.jsp) on the Healthcare Cost and Utilization Project (HCUP) website for more info on how to obtain the target ICD codes for different clinical conditions of interest. After obtaining the set of related ICD codes, we can then post queries 
with respect to the columns: `diagnosis_codes` and `billing_diagnosis_codes` to pull relevant rows from Andromeda (see **cohort_search.py** for 
example queries).

The clinical variables used to predict/correct LOINC codes are the columns/attributes of the table obtraind from applying transformations.withMedivoTestResultType() available from [Samantha](https://github.com/medivo/samantha/blob/master/src/main/scala/ai/prognos/samantha/clinical/transformations.scala). Note that useful variables for the prediction may be just a subset of these patient attributes. Example variables are: `test_result_name`, `test_result_value`, `test_order_name`, `test_result_units_of_measure`, among many others. Class labels for the training data are the LOINC codes (as they are what we are trying to predict). LOINC labels are avaiable through `test_result_loinc_code`. 

Due to the size limit, we will not share the full dataset here. However, coming up, we shall upload sample (toy) datasets ... 

**3. External, non-target-disease data**:

Similar to the input training data, "external data" may be gathered for balancing the sample sizes 
in the training data given in (2). A careful EDA will often indicate that a subset of LOINC codes have small sample sizes. To balance 
the class sample sizes as much as possible, it may be of interest to gather more training data from random subset of Andromeda matching 
our disease-specific LOINC codes. 

This external data from non-target disease cohort are also, by default, assumed to be kept under `<project_dir>/loinc_predictor/data`


**4. LOINC resources**: 

Relevant files such as **LoincTable.csv**, MapTo.csv are expected to be read from: `<project_dir>/loinc_predictor/LoincTable`


## Directory Structure

Each indivdual files include

```bash
loinc_predictor/
│
│
├── config.py (system-wise configiration file)
├── ClassifierArray.ipynb (main entry for the classifier array approach) 
├── MatchmakerPredictor.py (main entry for the matchmaking approach; NOT completed, please use the following two files for now)
│
├── feature_gen.py (main entry for feature generation required for the Matchmaker)
├── matchmaker_analyzer.py (a prototype for matchmaker; still under error analysis as the name suggested)     
│
├── analyzer.py (main analysis entry) 
├── ... 
│
├── data/ 
│   ├──  andromeda-pond-hepatitis-c.csv.fake (note that only this file is essential; the rest can generated)
│   ├──  andromeda-pond-hepatitis-c-processed.csv.fake (generated via CleanTextData.py)
│   ├──  andromeda-pond-hepatitis-c-balanced.csv.fake (generated via analyzer.py; may be time-consuming to generate)
│
├── LoincTable/   
│
├── result/
│   ├──  performance-hepatitis-c.csv
│   ...
│
├── doc/
│   ├── vars_analysis.txt
│   ├── vars_timestamp.txt
│   ├── vars_mtrt.txt
│
├── MANIFEST.in
├── README.md
└── setup.py
```

Input and Output
----------------
* Data loader(s) assume, by default, that the input data comes from a "data" directory
directly under the working/project directory (where all the python modules are kept)

    e.g. <project_dir>/loinc_predictor/data

* The Loinc table and its resources are by default read from the following directory: 

     <project_dir>/loinc_predictor/LoincTable

* Predictive performance evaluation is saved as a dataframe to the following directory 

     <project_dir>/loinc_predictor/result
     
