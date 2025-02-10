# Pipeline: Binary Classification for Patient Outcomes

This guide walks through the steps required to **finetune a model for binary classification** of patient outcomes. The pipeline consists of:

1. [**Create Data**](#1-create-data)
2. [**Pretrain**](#2-pretrain)
3. [**Create Outcome Definition**](#3-create-outcome-definition)
4. [**Define Study Cohort**](#4-define-study-cohort)
5. [**Finetune Model**](#5-finetune-model)
6. [**Out-of-Time Evaluation (Temporal Validation)**](#6-out-of-time-evaluation-temporal-validation)

---

## 1. Create Data

The starting point is a set of CSV files containing patient-level metadata and event-level data, consider using the [EHR_PREPROCESS](https://github.com/kirilklein/ehr_preprocess.git) repository to generate formatted data in CVS files.

For example, you might have:

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

## 2. Pretrain

The `pretrain` script trains a base BERT model on the tokenized medical data.

## 3. Create Outcome Definition

The `create_outcomes` script defines and extracts patient outcomes from structured data.

### Configuration

Edit the **outcomes configuration file**:

```yamlloader:
concepts: [
    diagnose # include all files that are needed for matching
  ]
match: ["D...", "M..."]  # Concepts to match, e.g. diagnoses, medications, procedures
match_how: "exact"       # Matching method (exact, contains, startswith)
case_sensitive: false    # Case sensitivity for matching
```

### Outputs

- A CSV file containing **outcome timestamps** for each patient.
- If needed, a separate outcome file can be created to serve as the **exposure**.

---

## 4. Define Study Cohort

The `select_cohort` script selects patients based on predefined criteria.

### Cohort Configuration

Edit the **cohort configuration file**:

```yaml
# Data Splitting
test_ratio: 0.2    # Proportion of data for testing
cv_folds: 5        # Number of cross-validation folds

# Patient Selection
selection:
  exclude_prior_outcomes: true  # Remove patients with prior outcomes
  exposed_only: false           # Include both exposed and unexposed patients

# Age Filters
age:
  min_years: 18
  max_years: 120

# Demographics
categories:
  GENDER:
    include: [M]  # Only include male patients
    # Alternative: exclude: [F]  # Exclude female patients

# Index Date Configuration
index_date:
  mode: relative  # 'relative' (to exposure) or 'absolute' (specific date)

  absolute:
    year: 2015
    month: 1
    day: 26

  relative:
    n_hours_from_exposure: -24  # Relative to exposure (-24 = 24h before)
```

### Cohort Outputs

- **`pids.pt`**: List of patient IDs
- **`index_dates.csv`**: Timestamps for patient-specific index dates
- **`folds.pt`**: Cross-validation fold assignments
- **`test_pids.pt`**: Test set patient IDs

---

## 5. Finetune Model

The `finetune_cv` script trains a model using the selected cohort.

### Finetuning Configuration

Edit the **training configuration file**:

```yaml
# Outcome Definitions
outcome:

  n_hours_censoring: -10        # Censor outcomes occurring this many hours before index date
  n_hours_start_follow_up: 1    # Start of outcome observation window
  n_hours_end_follow_up: 168    # End of observation window (e.g., 7 days), can be null

# Training Parameters
trainer_args:
  batch_size: 256
  epochs: 100
  early_stopping: 10            # Stop training if no improvement after 20 epochs
  stopping_criterion: roc_auc   # Performance metric to monitor
```

### Process

- The model is trained and validated on **cross-validation folds**.
- The best-performing checkpoint is saved.
- Finally, the model is **evaluated on the test set**.

---

## 6. Out-of-Time Evaluation (Temporal Validation)

To test generalization across time, models are evaluated on **data from different time periods**.

### Step 1: Create Temporal Splits

Use `select_cohort` to separate training and test sets based on time.

Example **absolute index date** for test data:

```yaml
absolute:
  year: 2015
  month: 1
  day: 26
```

### Step 2: Train Model with Temporal Constraints

Adjust **validation** and **test** windows:

```yaml
n_hours_censoring: 0        # Censor at index_date
n_hours_start_follow_up: 100    # Start 100 hours after index_date
n_hours_end_follow_up: 10_000    # End 10k hours after index_date
```

### Step 3: Validate Model on Future Data

Shift follow-up periods for evaluation:

```yaml
n_hours_censoring: 10_000    # Ignore outcomes before training cutoff
n_hours_start_follow_up: 10_100  # Start after training input period ends
n_hours_end_follow_up: null     # Extend follow-up indefinitely
```

---

## Summary

| Step                     | Script           | Key Configs | Output Files |
|--------------------------|-----------------|-------------|-------------|
| **1. Outcome Definition** | `create_outcomes` | Outcome matching criteria | `outcomes.csv` |
| **2. Cohort Selection** | `select_cohort` | Patient criteria, demographics, index date | `pids.pt`, `folds.pt`, `index_dates.csv` |
| **3. Model Finetuning** | `finetune_cv` | Censoring time, follow-up window, training params | Trained model, performance metrics |
| **4. Temporal Validation** | `select_cohort` + `finetune_cv` | Time-based validation, shifting follow-up | Evaluation results |

---
  
  📖 **A good starting point are the examples in the `configs` folder.**
