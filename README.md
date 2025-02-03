# COREBEHRT

[![Pipeline test](https://github.com/FGA-DIKU/EHR/actions/workflows/pipeline.yml/badge.svg)](https://github.com/FGA-DIKU/EHR/actions/workflows/pipeline.yml)
[![Unittests](https://github.com/FGA-DIKU/EHR/actions/workflows/unittests.yml/badge.svg)](https://github.com/FGA-DIKU/EHR/actions/workflows/unittests.yml)
[![Formatting using black](https://github.com/FGA-DIKU/EHR/actions/workflows/format.yml/badge.svg)](https://github.com/FGA-DIKU/EHR/actions/workflows/format.yml)
[![Lint using flake8](https://github.com/FGA-DIKU/EHR/actions/workflows/lint.yml/badge.svg)](https://github.com/FGA-DIKU/EHR/actions/workflows/lint.yml)

COREBEHRT is a framework for processing and analyzing Electronic Health Records (EHR) data using BERT-based models.

## Getting Started

### Virtual Environment Setup

For running tests and pipelines, create and activate a virtual environment, then install the required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
(.venv) pip install -r requirements.txt
```

## Unittests

### In Linux

Enable your virtual environment and run the unittests:

```bash
(.venv) python -m unittest
```

## Pipeline

The pipeline can be run from the root directory by executing the following commands:

```bash
(.venv) python -m corebehrt.main.create_data
(.venv) python -m corebehrt.main.pretrain
(.venv) python -m corebehrt.main.create_outcomes
(.venv) python -m corebehrt.main.finetune_cv
```

### 1. Create Data

Processes raw EHR data into tokenized features suitable for model training. This step:

- Converts raw medical concepts into numerical tokens
- Handles temporal information
- Creates patient segments
- Processes background variables (e.g., gender)

### 2. Pretrain

Trains the base BERT model on the tokenized medical data to learn general representations of medical concepts and their relationships. This unsupervised learning phase helps the model understand the underlying patterns in medical data.

### 3. Create Outcomes

Generates outcome labels from the formatted data for supervised learning:

- Processes specified medical conditions or events as outcomes
- Stores outcomes with absolute temporal positions
- Supports multiple outcome definitions

### 3.1 Create Cohort

Defines the study population by:

- Creating a filtered list of patient IDs based on inclusion/exclusion criteria
- Generating a table of index dates for temporal alignment
- Supporting customizable cohort selection criteria

### 4. Finetune

Adapts the pretrained model for specific prediction tasks:

- Uses supervised learning on the defined outcomes
- Supports cross-validation
- Allows for task-specific model optimization

## Data Processing

### Feature Creation

The `FeatureCreator` class transforms raw EHR data into structured features:

Input format:

| PID | CONCEPT | ADMISSION_ID | TIMESTAMP | ... |
|-----|---------|--------------|-----------|-----|

Combined with patient data:

| PID | GENDER | BIRTHDATE | DEATHDATE | ... |
|-----|---------|-----------|-----------|-----|

Produces:

| PID | concept | abspos | segment | age | ... |
|-----|---------|--------|---------|-----|-----|

## Azure Integration

For running COREBEHRT on Azure cloud infrastructure using SDK v2, refer to the [Azure guide](corebehrt/azure/README.md). This includes:

- Configuration setup for Azure
- Data store management
- Job execution in the cloud
- Environment preparation
