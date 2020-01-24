Introduction
------------

Standards and accuracy for the Logical Observation Identifiers Names and Codes (LOINC) are 
critical for interoperability and data sharing. In addition, many disease phenotyping 
analytics are also contingent upon the accuracy of the LOINC codes. However, there are
non-trivial instances of inconsistency and inaccuracy in the general EMR data such as 
Prognos's Andromeda datalake. Without a consistent coding accuracy, clinical data may not 
be easily harmonized, shared, or interpreted in a meaningful context. 

We seek to develop an automated pipeline using machine learning, NLP and rule-based methods 
that leverages noisy labels to map laboratory data to LOINC codes. 

The target LOINC codes for this module depends on the input condition of interest. For instance, 
patients with Hepatitis C are associated with a set of LOINC codes (700+), which 
represent all the known lab tests in different measurement units, standards, among others, 
that can be gleaned from Andromeda datalake (or your own source of data). 

For further details on how LOINC codes are structured. 

If the LOINC codes already exist for a particular row (of the patient data), this module can be 
thought of as providing a basis for validation; if on the other hand the LOINC codes do not
already exist for certain patient records, then this module can serve as a LOINC codes predictor. 

Prerequisite
------------
For ease of illustration, we shall assume that the package (loinc_predictor) is installed under 
<project_dir>/loinc_predictor, where project_dir is the directory of your choice hosting this module.


1. Non-standard modules: Please install the following dependent packages
   - [tabulate](https://pypi.org/project/tabulate/): Pretty print tabular data such as pandas dataframe
   - [feature_selector](https://pypi.org/project/feature-selector/): Used to identify and select important features for ML algorithms
   - [gensim](https://pypi.org/project/gensim/): A library for information retrival, topic modeling, string comparisons and other NLP tasks 

2. Input data:

Training LOINC predictive models requires input training data. These data are assumed to 
have been made available and kept under <project_dir>/loinc_predictor/data

The training data are curated based on the disease of interest. For instannce, a Hepatitis-C dataset would comprise 
sampled rows of patient data from Andromeda that match a set of ICD codes pertaiing to Hepatitis C. Please refer to 
[Clinica Classfication Software](https://www.hcup-us.ahrq.gov/tools_software.jsp) on Healthcare Cost and 
Utilization Project (HCUP) website for more info on how to obtain the target ICD codes for different clinical conditions of interest.

The data source came from subsampling Andromeda datalake. The clinical variables used to predict/correct LOINC codes 
are the columns/attributes of the table obtraind from applying transformations.withMedivoTestResultType() available from 
[Samantha](https://github.com/medivo/samantha/blob/master/src/main/scala/ai/prognos/samantha/clinical/transformations.scala). Note 
that useful variables for the prediction may be just a subset of these patient attributes. Example variables are: `test_result_name`, 
`test_result_value`, `test_order_name`, `test_result_units_of_measure`, among many others. Class labels for the training 
data are the LOINC codes (as they are what we are trying to predict). LOINC labels are avaiable through `test_result_loinc_code`. 

Coming up, we will upload sample datasets ... 

3. External, non-target-disease data:

Similar to the input training data, "external data" may be gathered for balancing the sample sizes 
in the training data given in (2). A careful EDA will show that a least a subset of LOINC This external data from non-target disease 
cohort are also, by default, assumed to be kept under <project_dir>/loinc_predictor/data

Note that the 

4. Loinc resources: 

Relevant files such as LoincTable.csv, MapTo.csv are kept under:

<project_dir>/loinc_predictor/LoincTable


## Directory Structure

```bash
loinc_predictor/
│
├── analyzer.py (main entry) 
├── Analyzer.ipynb (main entry notebook)
├── Learner.ipynb
├── ... 
│
├── data/ 
│
├── LoincTable/   
│
├── result/
│   ├── performance-hepatitis-c.csv
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
* Data loading functions assume, by default, that the input comes from a data directory
under the working directory e.g. <project_dir>/loinc_predictor/data

* Loinc tables and resources are by default read from the following directory: 

     <project_dir>/loinc_predictor/LoincTable

* Predictive performance evaluation is saved as a dataframe to the following directory 

     <project_dir>/loinc_predictor/result
     

