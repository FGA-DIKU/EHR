# EHR

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
Lists the step in the pipeline and how to run them.

### 1. Create Data
The pretraining process is handled by the [`main_create_data.py`](corebehrt/main_create_data.py) script.

Enable your virtual environment and run:
```
(.venv) $ python -m corebehrt.main_create_data
```
This creates the `outputs` folder.

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
