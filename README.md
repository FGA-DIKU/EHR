# COREBEHRT
[![Pipeline test](https://github.com/FGA-DIKU/EHR/actions/workflows/pipeline.yml/badge.svg)](https://github.com/FGA-DIKU/EHR/actions/workflows/pipeline.yml)
[![Unittests](https://github.com/FGA-DIKU/EHR/actions/workflows/unittests.yml/badge.svg)](https://github.com/FGA-DIKU/EHR/actions/workflows/unittests.yml)
[![Formatting using black](https://github.com/FGA-DIKU/EHR/actions/workflows/format.yml/badge.svg)](https://github.com/FGA-DIKU/EHR/actions/workflows/format.yml)
[![Lint using flake8](https://github.com/FGA-DIKU/EHR/actions/workflows/lint.yml/badge.svg)](https://github.com/FGA-DIKU/EHR/actions/workflows/lint.yml)


**COREBEHRT** 

## Virtual environment
For running the tests and pipelines, it is adviced to create a virtual environment, enable it, and install the requirements.
```
$ python -m venv .venv
$ source .venv/bin/activate
(.venv) $ pip install -r requirements.txt
```

## Unittests
### In Linux
Enable your virtual environment and run the unittests:
```
(.venv) $ python -m unittest
```

## Pipeline
The pipeline can be run from the root directory by executing the following commands:
```
(.venv) $ python -m corebehrt.main.create_data
(.venv) $ python -m corebehrt.main.pretrain
(.venv) $ python -m corebehrt.main.create_outcomes
(.venv) $ python -m corebehrt.main.finetune_cv
```

### 1. Create Data
Creates tokenized features from the formatted data.

### 2. Pretrain
Pretrains the model on the tokenized features.

### 3. Create Outcomes
Creates the outcomes from the formatted data.
Outcomes are stored as absolute positions.

### 3.1 Create Cohort
Creates a cohort from the formatted data.
Cohort is stored as a list of PIDs and a table of index_dates

### 4. Finetune
Finetunes the pretrained model on the outcomes.


## Classes
#### [`FeatureCreator`](corebehrt/classes/features.py)
From the **raw data**
PID|CONCEPT|ADMISSION_ID|TIMESTAMP|...
---|-------|------------|---------|---

and **patient data**

PID|GENDER|BIRTHDATE|DEATHDATE|...
---|------|---------|---------|---

we create: 

PID|concept|abspos|segment|age|...
---|-------|------|-------|---|---

and include the following:
- background 
- death event

#### [`Excluder`](corebehrt/classes/excluder.py)
- incorrect ages
- nans
- normalize segments after exclusion
- short sequences

Results are saved in a table.

#### [`EHRTokenizer`](corebehrt/classes/tokenizer.py)
Currently, still operates on sequences.
Adds SEP and CLS tokens
Create vocabulary based on pretrain_data


## Azure
Use the submodule `corebehrt.azure` for running on Azure with SDK v2. See [how-to](corebehrt/azure/README.md).
