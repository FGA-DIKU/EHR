# Detailed Functional Structure

## Cohort Handling

### `matching.py`

String matching utilities for patient data.

- **`get_col_booleans`**: Get boolean columns for pattern matching
- **`startswith_match`**: Match strings using startswith
- **`contains_match`**: Match strings using contains

### `outcomes.py`

Binary outcome generation from patient data.

- **`get_binary_outcomes`**: Create binary outcomes for each patient
- Handles follow-up windows and temporal alignment

## Features

### `creators.py`

Feature creation utilities.

- **`create_abspos`**: Create absolute positions from timestamps
- **`create_age_in_years`**: Compute patient ages
- **`create_death`**: Create death events
- **`create_background`**: Create background concepts
- **`create_segments`**: Assign segments to concepts

### `exclude.py`

Data exclusion utilities.

- **`exclude_incorrect_event_ages`**: Filter events by age range
- **`exclude_event_nans`**: Remove events with missing values

### `normalize.py`

Normalization utilities.

- **`min_max_normalize`**: Perform min-max normalization
- **`normalize_segments_for_patient`**: Normalize patient segments
- **`normalize_segments`**: Make segment IDs zero-based and contiguous

### `split.py`

Dataset splitting utilities.

- **`split_pids_into_pt_ft_test`**: Split into pretrain/finetune/test
- **`split_pids_into_train_val`**: Split into train/validation
- **`get_n_splits_cv_pids`**: Create cross-validation splits

### `tokenize.py`

Tokenization utilities.

- **`add_special_tokens_partition`**: Add CLS/SEP tokens
- **`tokenize_partition`**: Convert concepts to token IDs
- **`limit_concept_length_partition`**: Truncate concept strings

## IO Operations

### `load.py`

Data loading utilities.

- **`load_concept`**: Load concept data
- **`load_vocabulary`**: Load tokenizer vocabulary

### `save.py`

Data saving utilities.

- **`save_pids_splits`**: Save train/val splits
- **`save_vocabulary`**: Save tokenizer vocabulary

## Preparation

### `convert.py`

Data conversion utilities.

- **`dataframe_to_patient_list`**: Convert DataFrame to PatientData objects

### `filter.py`

Data filtering utilities.

- **`filter_table_by_pids`**: Filter by patient IDs
- **`remove_missing_timestamps`**: Remove invalid timestamps
- **`exclude_short_sequences`**: Remove short sequences
- **`censor_patient`**: Apply temporal censoring

### `truncate.py`

Sequence truncation utilities.

- **`truncate_patient`**: Truncate patient sequences
- **`standard_truncate_patient`**: Basic truncation strategy
- **`prioritized_truncate_patient`**: Priority-based truncation

### `utils.py`

General preparation utilities.

- **`get_background_length`**: Get background sequence length
- **`get_hours_since_epoch`**: Calculate temporal positions
- **`get_non_priority_tokens`**: Identify low-priority tokens

## Setup

### `args.py`

Argument parsing utilities.

- **`get_args`**: Parse command line arguments

### `checks.py`

Validation utilities.

- **`check_categories`**: Validate category rules
- **`check_concepts_columns`**: Validate concept data
- **`check_patients_info_columns`**: Validate patient info

### `model.py`

Model setup utilities.

- **`get_last_checkpoint_epoch`**: Find latest checkpoint
- **`load_model_cfg_from_checkpoint`**: Load model config

## Trainer

### `collate.py`

Batch collation utilities.

- **`dynamic_padding`**: Pad sequences in batch

### `setup.py`

Training setup utilities.

- **`replace_steps_with_epochs`**: Convert epoch counts to steps
- **`convert_epochs_to_steps`**: Calculate training steps
