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

Our pipeline simulates a real-world deployment scenario by distinguishing the data available for training from that used during testing.

### Out-of-Time Evaluation with Absolute Index Dates

1. **Fixed Reference Date & Censoring:**  
   All patients are assigned an absolute index date (e.g., January 1, 2020). This date serves as the reference for training, though it isn’t necessarily the last available date since we may censor outcomes relative to it (using the `n_hours_censoring` parameter, however for absolute index dates it makes most sense to set `n_hours_censoring` to 0).

2. **Cohort Splitting After Index Date Creation:**  
   Once index dates are computed (and any censoring logic is in place), the cohort is split into training/validation and test sets. This ensures the split reflects the final, fully defined cohort.

3. **Test Shift to Simulate Future Prediction:**  
   To mimic a scenario where the model is trained with data up to the cutoff but then deployed later, we apply a shift (using `test_shift_hours`) exclusively to test patients. For example, with a 1 year test shift, training is performed using data up to January 1, 2020 (with outcomes censored relative to that date), while test patients are assigned a shifted index date of January 1, 2021. This simulates that the model is being applied to predict outcomes in a future time period.

4. **Follow-up Window:**
   The follow-up window is defined by the `n_hours_start_follow_up` and `n_hours_end_follow_up` parameters. For example, with a 1 year follow-up window, the model will predict outcomes in the period from January 1, 2020 to January 1, 2021 for train patients and from January 1, 2021 to January 1, 2022 for test patients.

**Example Configuration:**

With this example config we fine-tune the model using data available up to 01/01/2020, predicting outcomes from 01/01/2020-01/01/2021. For testing, we use data up to 01/01/2021 to predict outcomes from 01/01/2021-01/01/2022.

In select cohort:

- **Absolute Index Date:** January 1, 2020  
- **test_shift_hours:** 365 * 24 (1 year)

Fine-tuning configuration (outcome):

- **n_hours_censoring:** 0 (censor outcomes occurring within 24 hours before the index date)  
- **n_hours_start_follow_up:** 0
- **n_hours_end_follow_up:** 365 * 24 (1 year)

**Process Overview:**

- All patients are assigned an index date of January 1, 2020.
- The cohort is split into training/validation and test sets after index date creation.
- For test patients, the index date is shifted by one year (to January 1, 2021), so that:
  - The model is trained using input data available up to January 1, 2020.
  - Outcomes for training/validation are observed from January 1, 2020 to January 1, 2021, while outcomes for testing are observed from January 1, 2021 to January 1, 2022.

This approach ensures that our evaluation mimics prospective deployment, where the model’s training and testing data reflect distinct time periods.

### Step-by-Step Process

#### Step 1: Pretrain Model on cutoff data

Use the cutoff date option in the pretrain config to pretrain a model on data from a specific time period.
E.g.

```yaml
data:
  cutoff_date:
    year: 2020
    month: 1
    day: 1
```

#### Step 2: Create Temporal Splits

Use `select_cohort` to define the study cohort with absolute index dates and then split the cohort into training/validation and test sets.

- All patients are assigned an absolute index date (e.g., January 1, 2020).
Example **absolute index date** for test data:
- A test shift is applied to test patients using test_shift_hours (e.g., 365 * 24 for a 1-year shift).
  - Training/Validation: Retain the original index date (e.g., January 1, 2020).
  - Test: The index date is shifted (e.g., to January 1, 2021), simulating that predictions are made on data from a later time period.
Example Configuration:

```yaml
absolute:
  year: 2020
  month: 1
  day: 1
test_shift_hours: 365 * 24
```

#### Step 3: Train Model with Temporal Constraints

For training, use the input data defined relative to the original index date.
For example, set:

```yaml
n_hours_censoring: 0 
n_hours_start_follow_up: 0   
n_hours_end_follow_up: 365 * 24    
```

This configuration ensures that the model is trained on data available up to the fixed cutoff date, while test patients receive a shifted index date (e.g., shifted by 1 year) that defines their outcome follow-up window. This effectively simulates a future prediction scenario.

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
