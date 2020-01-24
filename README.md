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
For ease of illustration, we will now assume that the package (loinc_predictor) is installed 
under <project_dir>/loinc_predictor such that all the related files are directly under loinc_predictor


1. Non-standard modules: 

2. Input data:

Training LOINC predictive models requires input training data. These data are assumed to 
have been kept under <project_dir>/loinc_predictor/data

3. External data used to balance classes: 

Similar to the input training data, extra training data gathered from non-target disease 
cohort are also, by default, assumed to the kept under <project_dir>/loinc_predictor/data


4. Loinc resources: 

Relevant files such as LoincTable.csv, MapTo.csv are kept under:

<project_dir>/loinc_predictor/LoincTable


Directory Structure
-------------------

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
     

