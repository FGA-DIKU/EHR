# CoreBEHRT Functional Overview

## Overview by Submodule

### Cohort Handling

Functions for patient cohort management:

- Patient matching and filtering (`matching.py`)
- Outcome generation and processing (`outcomes.py`)
- Temporal alignment of patient data
- Cohort selection criteria implementation

### Features

Feature processing utilities:

- Feature creation and transformation (`creators.py`)
- Data exclusion logic (`exclude.py`)
- Value normalization (`normalize.py`)
- Tokenization utilities (`tokenize.py`)
- Data splitting functionality (`split.py`)

### IO Operations

Data input/output management:

- Data loading utilities (`load.py`)
- Save operations (`save.py`)
- File format handling
- Data validation and checks

### Preparation

Data preparation utilities:

- Data conversion functions (`convert.py`)
- Patient data filtering (`filter.py`)
- Sequence truncation logic (`truncate.py`)
- Helper utilities (`utils.py`)

### Setup

Configuration and initialization:

- Argument parsing (`args.py`)
- Validation checks (`checks.py`)
- Model configuration (`model.py`)
- Environment setup

### Trainer

Training-related utilities:

- Batch collation (`collate.py`)
- Training setup helpers (`setup.py`)
- Training configuration
- Batch processing utilities

## Key Features

- Pure functions for data manipulation
- Utility functions supporting module operations
- Data validation and transformation
- Configuration and setup helpers
- Training support functions
