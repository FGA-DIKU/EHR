# COREBEHRT

[![Pipeline test](https://github.com/FGA-DIKU/EHR/actions/workflows/pipeline.yml/badge.svg)](https://github.com/FGA-DIKU/EHR/actions/workflows/pipeline.yml)
[![Unittests](https://github.com/FGA-DIKU/EHR/actions/workflows/unittests.yml/badge.svg)](https://github.com/FGA-DIKU/EHR/actions/workflows/unittests.yml)
[![Formatting using black](https://github.com/FGA-DIKU/EHR/actions/workflows/format.yml/badge.svg)](https://github.com/FGA-DIKU/EHR/actions/workflows/format.yml)
[![Lint using flake8](https://github.com/FGA-DIKU/EHR/actions/workflows/lint.yml/badge.svg)](https://github.com/FGA-DIKU/EHR/actions/workflows/lint.yml)

> **A framework for processing and analyzing Electronic Health Records (EHR) data using BERT-based models.**

COREBEHRT helps researchers and data scientists preprocess EHR data, train models, and generate outcomes for downstream clinical predictions and analyses.

---

## Table of Contents

- [COREBEHRT](#corebehrt)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [Directory Overview](#directory-overview)
  - [Getting Started](#getting-started)
    - [Virtual Environment Setup](#virtual-environment-setup)
  - [Pipeline](#pipeline)
    - [1. Create Data](#1-create-data)
    - [2. Pretrain](#2-pretrain)
    - [3. Create Outcomes](#3-create-outcomes)
    - [3.1 Create Cohort](#31-create-cohort)
    - [4. Finetune](#4-finetune)
      - [Out-of-Time Evaluation](#out-of-time-evaluation)
  - [Azure Integration](#azure-integration)
  - [Contributing](#contributing)
  - [License](#license)
  - [Citation](#citation)

---

## Key Features

- **End-to-end EHR Pipeline**: Tools for data ingestion, cleaning, and feature extraction.
- **BERT-based Modeling**: Pretraining on massive EHR corpora followed by task-specific finetuning.
- **Cohort Management**: Flexible inclusion/exclusion logic, temporal alignment, outcome definition.
- **Scalable**: Designed to run both locally or on cloud infrastructure (Azure).
- **Built-in Validation**: Cross-validation and out-of-time evaluation strategies.

---

## Directory Overview

Below is a high-level overview of the most important directories:

- **main**: Primary pipeline scripts (create_data, pretrain, finetune, etc.)
- **modules**: Core implementation of model architecture and data processing ([detailed overview](corebehrt/modules/overview.md))
- **configs**: YAML configuration files for each pipeline stage
- **functional**: Pure utility functions supporting module operations ([detailed overview](corebehrt/functional/overview.md))
- **azure**: Cloud deployment and execution utilities ([azure instructions](corebehrt/azure/README.md))

## Getting Started

### Virtual Environment Setup

For running tests and pipelines, create and activate a virtual environment, then install the required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
(.venv) pip install -r requirements.txt
```

## Pipeline

The pipeline can be run from the root directory by executing the following commands:

```bash
(.venv) python -m corebehrt.main.create_data
(.venv) python -m corebehrt.main.pretrain
(.venv) python -m corebehrt.main.create_outcomes
(.venv) python -m corebehrt.main.create_cohort
(.venv) python -m corebehrt.main.finetune_cv
(.venv) python -m corebehrt.main.evaluate_cv # not implemented yet
```

### 1. Create Data

This step converts **raw EHR data** into **tokenized features** suitable for model training. The core tasks include:

- **Vocabulary Mapping**: Translates raw medical concepts (e.g., diagnoses, procedures) into numerical tokens.
- **Temporal Alignment**: Converts timestamps into relative positions (e.g., hours or days from an index date).
- **Patient Segmentation**: Splits patients into fixed-length segments for model input.
- **Background Variables**: Incorporates static features such as age, gender, or other demographics.
- **Efficient Output**: Produces a structured binary or parquet format that can be rapidly loaded in subsequent steps.

If you need a starting point for raw EHR data formatting, consider using the [EHR_PREPROCESS](https://github.com/kirilklein/ehr_preprocess.git) repository to generate pre-cleaned CSV files. For example, you might have:

- **`patients_info.csv`** (holding patient-level metadata)

  | PID  | BIRTHDATE   | DEATHDATE   | GENDER | …   |
  |------|-------------|-------------|--------|-----|
  | p1   | 1980-01-01  | 2022-06-01  | M      | …   |
  | p2   | 1975-03-12  | NaN         | F      | …   |
  | …    | …           | …           | …      | …   |

- **`concept.diagnose.csv`, `concept.procedure.csv`, `concept.medication.csv`, `concept.lab.csv`, …** (event-level data)

  | TIMESTAMP   | PID  | ADMISSION_ID | CONCEPT  | …   |
  |-------------|------|--------------|----------|-----|
  | 2012-01-05  | p1   | adm101       | D123     | …   |
  | 2012-01-05  | p1   | adm101       | D124     | …   |
  | 2015-03-20  | p2   | adm205       | D786     | …   |
  | …           | …    | …            | …        | …   |

These files feed into the **Create Data** pipeline, which merges, tokenizes, and structures the data for subsequent **Pretrain** and **Finetune** steps.

### 2. Pretrain

Trains the base BERT model on the tokenized medical data:

- Uses masked language modeling to learn contextual representations
- Processes large volumes of unlabeled EHR sequences
- Captures temporal and semantic relationships between medical concepts
- Saves checkpoints for downstream finetuning
- Supports distributed training on multiple GPUs

### 3. Create Outcomes

Generates outcome labels from the formatted data for supervised learning:

- Processes specified medical conditions or events as binary outcomes
- Handles time-to-event data with censoring
- Stores outcomes aligned with absolute temporal positions
- Supports multiple concurrent outcome definitions
- Includes validation checks for outcome integrity

### 3.1 Create Cohort

Defines the study population by:

- Applying inclusion/exclusion criteria based on diagnoses, procedures, or demographics
- Generating index dates for temporal alignment of patient trajectories
- Supporting both point-in-time and period-based cohort definitions
- Enabling stratified sampling of sub-populations
- Maintaining train/validation/test splits

### 4. Finetune

Adapts the pretrained model for specific prediction tasks:

- Performs supervised learning on defined binary outcomes
- Implements k-fold cross-validation with stratification
- Supports both classification and time-to-event prediction
- Enables transfer learning from pretrained checkpoints
- Includes early stopping and model selection based on validation metrics
- Allows specification of excluded patient IDs for held-out test sets

#### Out-of-Time Evaluation

To perform temporal validation (out-of-time evaluation) of your models, follow these steps:

1. **Split Data**
   - Use `select_cohort` to create separate test and cross-validation sets
   - Use absolute index date:
   - Example configuration:

     ```yaml
     absolute:
       year: 2015
       month: 1
       day: 26
     ```

2. **Train Model**
   - For out of time training (not implemented yet)
     - Set outcomes_val to have a shifted censoring/start follow up/end follow up for validation set
   - During cross-validation training, set an end-of-follow-up date
   - This date acts as a temporal cutoff, ignoring all outcomes that occur after it
   - Example configuration:
  
     ```yaml
     n_hours_censoring: 0        # censor at index_date
     n_hours_start_follow_up: 100    # start 100 hours after index_date
     n_hours_end_follow_up: 10000    # end 10k hours after index_date
     ```

3. **Validate Model**
   - Run validation on the test set using the validation script
   - Move censoring shift and start follow-up as needed, e.g. after the training cutoff date
   - Example configuration:
  
     ```yaml
     n_hours_censoring: 10000        # censor till end of train cutoff
     n_hours_start_follow_up: 10100  # start 10k+100 hours after train cutoff
     n_hours_end_follow_up: null     # end follow up at the end of test set
     ```

## Azure Integration

For running COREBEHRT on Azure cloud infrastructure using SDK v2, refer to the [Azure guide](corebehrt/azure/README.md). This includes:

- Configuration setup for Azure
- Data store management
- Job execution in the cloud
- Environment preparation

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code style and formatting
- Testing requirements
- Pull request process
- Issue reporting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use COREBEHRT in your research, please cite the following paper:

```bibtex
```
