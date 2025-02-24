# CoreBEHRT Configuration Files Overview
This repository contains configuration files for processing Electronic Health Record (EHR) data using CoreBEHRT. Below is an overview of the key configuration files used in different stages of data processing and modeling.

## Configuration Files Summary

| **Configuration File**       | **Purpose**                                              | **Key Functions** |
|-----------------------------|----------------------------------------------------------|-------------------|
| `create_data.yaml`          | Prepares structured EHR data for modeling               | Extracts clinical concepts (diagnoses, medications, procedures, lab tests), tokenizes records, normalizes values, and splits datasets into pretrain, finetune, and test sets. |
| `pretrain.yaml`             | Pretrains a transformer-based model using Masked Language Modeling (MLM) | Applies token masking, sequence truncation, and optimizes model parameters using Adam optimizer. Evaluates performance using MLM loss and precision metrics. |
| `outcome.yaml`              | Extracts clinical outcome labels for downstream modeling | Efficiently loads diagnosis-related data, applies filtering based on predefined criteria, and saves outcomes for further analysis. |
| `select_cohort.yaml`        | Selects a subset of patients based on predefined criteria | Filters patients based on age, gender, and exposure history, defines an index date, and splits data into training, validation, and test sets. |
| `fine_tune.yaml`            | Fine-tunes the pretrained model on clinical outcome prediction | Loads pretrained model, trains a classifier (GRU-based), converts outcome labels into binary classification, and optimizes training with gradient clipping and early stopping. |
| `finetune_evaluate.yaml`    | Evaluates the fine-tuned model's predictive performance | Computes metrics such as accuracy, precision, recall, ROC-AUC, PR-AUC, and tracks false/true positive and negative predictions. |

In the following sections, each configuration file is explained in detail.

## **Create Data (`create_data.yaml`)**  
This step **loads and processes raw EHR data**, extracts key clinical concepts, tokenizes records, and prepares structured inputs for modeling.  

### **Key Functions:**  
- Loads and processes **raw EHR data**, extracting **diagnoses, medications, procedures and lab tests**.  
- Defines **data paths** for raw data, tokenized sequences, and extracted features.  
- Tokenizes **patient records** into structured sequences for modeling.  
- Extracts **background variables** and sets a **reference timestamp** .  
- Configures **value processing**, including:  
  - **Binning** values into categories.  
  - **Normalization** for feature scaling.  
  - **Handling missing values**.  
- Splits the dataset into: **pretrain, finetune and test sets**  

### Hyperparameters for the `create data` stage:


| **Category**   | **Parameter**                   | **Value** |
|--------------|--------------------------------|----------|
| **Loader**   | `concept_types`               | `["diagnose", "medication", "procedure", "labtest"]` |
|              | `include_values`              | `["labtest"]` |
| **Features** | `background_vars`             | `["GENDER"]` |
|              | `origin_point`                | `2020-01-26` |
| **Values**   | `value_type`                  | `binned` |
|              | `normalize.func`              | `min_max_normalize_results` |
|              | `normalize.kwargs.min_count`  | `3` |
| **Excluder** | `min_age`                     | `-1` |
|              | `max_age`                     | `120` |
| **Split Ratios** | `pretrain`                 | `0.72` |
|              | `finetune`                    | `0.18` |

(For shared parameters, refer to [Common Hyperparameters](#common-hyperparameters-shared-across-all-stages))_
---

### Pretrain (`pretrain.yaml`)    
This step **trains a transformer-based model** on **EHR sequences** using **masked language modeling (MLM)** to learn meaningful patient data representations.  

#### **Key Functions:**  
- Loads **tokenized patient records** and **structured features** as inputs.  
- Applies **80% masking** and **10% token replacement** during MLM training.  
- Uses a **truncation length* and filters out sequences **shorter than minimum tokens**.  
- Splits data into:  
  - **training**  
  - **validation**  
- Trains for **number of epochs** 
- Optimizes using **Adam optimizer** 
- Saves pretrained models to `./outputs/pretraining/`.  
- Monitors performance using **top-1/top-10 precision** and **MLM loss**.  

| **Category**     | **Parameter**                        | **Value** |
|-----------------|-------------------------------------|----------|
| **Data**        | `dataset.select_ratio`                     | `1.0` |
|                 | `dataset.masking_ratio`                    | `0.8` |
|                 | `dataset.replace_ratio`                    | `0.1` |
|                 | `dataset.ignore_special_tokens`            | `true` |
|                 | `truncation_len`                   | `20` |
|                 | `val_ratio`                        | `0.2` |
|                 | `min_len`                          | `2` |
| **trainer_args**| `batch_size`                       | `32` |
|                 | `effective_batch_size`             | `64` |
|                 | `epochs`                           | `5` |
|                 | `gradient_clip.clip_value`        | `1.0` |
|                 | `shuffle`                          | `true` |
| **Optimizer**   | `lr`                               | `5e-4` |
|                 | `eps`                              | `1e-6` |
| **Scheduler**   | `_target_`                         | `transformers.get_linear_schedule_with_warmup` |
|                 | `num_warmup_epochs`               | `2` |
|                 | `num_training_epochs`             | `3` |
| **Metrics**     | `top1._target_`                    | `corebehrt.modules.monitoring.metrics.PrecisionAtK` |
|                 | `top1.topk`                        | `1` |
|                 | `top10._target_`                   | `corebehrt.modules.monitoring.metrics.PrecisionAtK` |
|                 | `top10.topk`                       | `10` |
|                 | `mlm_loss._target_`                | `corebehrt.modules.monitoring.metrics.LossAccessor` |
|                 | `mlm_loss.loss_name`               | `loss` |

---
### Define Outcomes (`outcome.yaml`)
This step **extracts and processes clinical outcome labels** from EHR records to support downstream modeling. The process is designed for efficiency, handling large volumes of diagnosis-related concepts while ensuring accurate filtering and classification.

#### Key Functions

##### Efficient Data Loading  
- Loads diagnosis-related concepts in **optimized batches** to enhance processing speed.  
- Reads data in **manageable chunks** for efficient handling.  

##### Outcome Labeling  
- **Defines `TEST_OUTCOME`**:  
  - Identifies specific clinical concepts but **excludes certain values** based on predefined criteria.  
  - Uses **case-sensitive matching** with a **"contains" filter** for concept identification.  
- **Defines `TEST_CENSOR`**:  
  - Matches relevant concepts using **case-insensitive "startswith" filtering**.  

##### Output Storage  
- Saves processed **outcome labels** to `./outputs/outcomes/`.  

### Hyperparameters for `outcome`

| **Parameter**                   | **Description**                                        | **Value** |
|----------------------------------|--------------------------------------------------------|----------|
| `paths.outcomes`                 | Directory to store processed outcomes                 | `./outputs/outcomes` |
| `loader.concepts`                | Concepts to extract                                  | `["diagnose"]` |
| `loader.batchsize`               | Number of records processed per batch               | `Optimized batch size` |
| `loader.chunksize`               | Number of records read at once                      | `Optimized chunk size` |
| `outcomes.TEST_OUTCOME.type`     | Outcome type                                        | `["CONCEPT"]` |
| `outcomes.TEST_OUTCOME.match`    | Concepts to include                                | `Predefined set` |
| `outcomes.TEST_OUTCOME.exclude`  | Concepts to exclude                                | `Predefined exclusions` |
| `outcomes.TEST_OUTCOME.match_how` | Matching method                                    | `contains` |
| `outcomes.TEST_OUTCOME.case_sensitive` | Case-sensitive matching                      | `true` |
| `outcomes.TEST_CENSOR.type`      | Censoring outcome type                             | `["CONCEPT"]` |
| `outcomes.TEST_CENSOR.match`     | Concepts included for censoring                   | `Predefined set` |
| `outcomes.TEST_CENSOR.match_how` | Censoring match method                            | `startswith` |
| `outcomes.TEST_CENSOR.case_sensitive` | Case-sensitive censoring matching              | `false` |

## Notes  
- The batch size and chunk size are **optimized for efficiency** without compromising accuracy.  
- `TEST_OUTCOME` and `TEST_CENSOR` use **different matching rules** to ensure precise classification.  
- Processed outcomes are **saved to `./outputs/outcomes/`** for further analysis.  
---

### **Hyperparameters for `select_cohort`**  

This configuration **selects a subset of patients** based on predefined criteria for further analysis.  

#### **Key Functions**  

##### Patient Data Loading  
- Loads **patient information** from a structured dataset.  
- Uses **exposure status**, if available, to refine patient selection.  

##### Patient Filtering  
- **Age criteria**: Includes only patients within a specific age range.  
- **Gender selection**: Filters the dataset based on a predefined gender category.  
- **Outcome history**: Excludes patients who had the outcome **before the index date**.  

##### Defining the Index Date  
- **Absolute reference**: Uses a fixed historical date.  
- **Relative reference**: Adjusts based on the first recorded exposure.  

##### Dataset Splitting  
- Divides the selected cohort into **training, validation, and testing** sets.  

##### Output Storage  
- Saves the final **cohort data** to a specified directory for further processing.  

---

#### **Hyperparameters for `select_cohort`**  

| **Parameter**                         | **Description**                                           | **Value** |
|---------------------------------------|-----------------------------------------------------------|----------|
| `paths.patients_info`                 | Path to patient information file                        | `./example_data/example_data_w_labs/patients_info.csv` |
| `paths.initial_pids`                  | Initial patient IDs (optional)                          | `./outputs/tokenized/pids_finetune.pt` |
| `paths.exposure`                       | Path to exposure file (optional)                        | `./outputs/outcomes/TEST_CENSOR.csv` |
| `paths.outcome`                        | Path to outcome file                                    | `./outputs/outcomes/TEST_OUTCOME.csv` |
| `paths.cohort`                         | Output directory for cohort data                        | `./outputs/cohort/` |
| `selection.exclude_prior_outcomes`     | Exclude patients with prior outcomes                    | `true` |
| `selection.exposed_only`               | Include only exposed patients                           | `false` |
| `selection.age.min_years`              | Minimum age for inclusion                               | `Configured limit` |
| `selection.age.max_years`              | Maximum age for inclusion                               | `Configured limit` |
| `selection.categories.GENDER.include`  | Gender selection criteria                               | `Predefined category` |
| `index_date.mode`                      | Index date mode (absolute/relative)                     | `relative` |
| `index_date.absolute.year`             | Absolute reference year                                 | `Configured date` |
| `index_date.absolute.month`            | Absolute reference month                                | `Configured date` |
| `index_date.absolute.day`              | Absolute reference day                                  | `Configured date` |
| `index_date.relative.n_hours_from_exposure` | Time offset from first exposure                  | `Configured offset` |
| `split_ratios.train`                   | Proportion of data allocated for training               | `Configured percentage` |
| `split_ratios.val`                     | Proportion of data allocated for validation             | `Configured percentage` |
| `split_ratios.test`                    | Proportion of data allocated for testing                | `Configured percentage` |

---
### **Fine-Tune & Evaluate (`fine_tune` & `finetune_evaluate`)**  

This phase **fine-tunes** the pretrained model on specific clinical outcomes and **evaluates** its performance using various metrics.  

### **Key Functions**  

#####  Fine-Tuning (`fine_tune`)  
- Loads a **pretrained model** from the designated directory.  
- Uses **tokenized data, extracted features, and cohort selection** for training.  
- **Trains a classifier (`ClassifierGRU`)** in a **bidirectional** mode.  
- Converts **outcome labels to binary values** based on their presence in a follow-up window.  
- Applies **sequence truncation** and removes extremely short sequences.  
- Splits data into **training and validation sets**.  
- Saves the **fine-tuned model** to the designated output directory.  

#####  Evaluation (`finetune_evaluate`)  
- Loads the **fine-tuned model** from the output directory.  
- Runs evaluation on a **test dataset** to assess predictive performance.  
- Computes **multiple evaluation metrics**, including:  
  - **Accuracy & Balanced Accuracy**  
  - **Precision & Recall**  
  - **ROC-AUC & PR-AUC**  
  - **True/False Positives & Negatives**  
  - **Mean Probability & Percentage of Positives**  

---

## **Hyperparameters for `fine_tune` & `finetune_evaluate`**  

_(For shared parameters, refer to [Common Hyperparameters](#common-hyperparameters))_  

| **Parameter**                        | **Description**                                          | **Value** |
|--------------------------------------|----------------------------------------------------------|----------|
| `paths.pretrain_model`               | Path to pretrained model                               | `./outputs/pretraining/` |
| `paths.outcome`                      | Outcome file location                                  | `./outputs/outcomes/TEST_OUTCOME.csv` |
| `paths.model`                        | Path to save fine-tuned model                         | `./outputs/finetuning/` |
| `model.cls._target_`                 | Model architecture (ClassifierGRU)                    | `ehr2vec.model.heads.ClassifierGRU` |
| `model.cls.bidirectional`            | Whether the GRU model is bidirectional                | `true` |
| `cv_splits`                          | Number of cross-validation splits                     | `Configured value` |
| `data.val_split`                     | Validation set percentage                             | `Configured percentage` |
| `data.truncation_len`                | Maximum sequence length                               | `Configured limit` |
| `data.min_len`                       | Minimum sequence length                               | `Configured limit` |
| `outcome.n_hours_censoring`          | Hours to censor after index date                     | `Configured offset` |
| `outcome.n_hours_start_follow_up`    | Start of follow-up period                            | `Configured time` |
| `outcome.n_hours_end_follow_up`      | End of follow-up period                              | `Configured time` |
| `trainer_args.val_batch_size`        | Validation batch size                                | `Configured size` |
| `trainer_args.effective_batch_size`  | Effective batch size                                 | `Configured size` |
| `trainer_args.gradient_clip.clip_value` | Gradient clipping value                          | `Configured value` |
| `trainer_args.shuffle`               | Shuffle training data                               | `true` |
| `trainer_args.early_stopping`        | Early stopping patience                             | `Configured patience` |
| `trainer_args.stopping_criterion`    | Stopping criterion                                  | `roc_auc` |
| `optimizer.eps`                      | Epsilon value for Adam optimizer                    | `Configured value` |
| `scheduler._target_`                 | Learning rate scheduler                            | `transformers.get_linear_schedule_with_warmup` |
| `scheduler.num_warmup_steps`         | Number of warmup steps                              | `Configured steps` |
| `scheduler.num_training_steps`       | Total number of training steps                      | `Configured steps` |
| `metrics.accuracy._target_`          | Accuracy metric                                    | `corebehrt.modules.monitoring.metrics.Accuracy` |
| `metrics.accuracy.threshold`         | Accuracy threshold                                 | `Configured threshold` |
| `metrics.roc_auc._target_`           | ROC-AUC metric                                    | `corebehrt.modules.monitoring.metrics.ROC_AUC` |
| `metrics.pr_auc._target_`            | PR-AUC metric                                     | `corebehrt.modules.monitoring.metrics.PR_AUC` |
| `metrics.precision._target_`         | Precision metric                                  | `corebehrt.modules.monitoring.metrics.Precision` |
| `metrics.recall._target_`            | Recall metric                                     | `corebehrt.modules.monitoring.metrics.Recall` |
| `metrics.mean_probability._target_`  | Mean probability metric                           | `corebehrt.modules.monitoring.metrics.Mean_Probability` |
| `metrics.percentage_positives._target_` | Percentage of positive predictions              | `corebehrt.modules.monitoring.metrics.Percentage_Positives` |
| `metrics.true_positives._target_`    | True Positives metric                             | `corebehrt.modules.monitoring.metrics.True_Positives` |
| `metrics.true_negatives._target_`    | True Negatives metric                             | `corebehrt.modules.monitoring.metrics.True_Negatives` |
| `metrics.false_positives._target_`   | False Positives metric                            | `corebehrt.modules.monitoring.metrics.False_Positives` |
| `metrics.false_negatives._target_`   | False Negatives metric                            | `corebehrt.modules.monitoring.metrics.False_Negatives` |

---

## Common Items
## **Common hyperparameters (Shared Across All Stages)**  

| **Parameter**              | **Description**                                      | **Value** |
|----------------------------|------------------------------------------------------|----------|
| `logging.level`            | Log verbosity level                                  | `INFO`   |
| `logging.path`             | Path to save logs                                    | `./outputs/logs` |
| `paths.features`           | Extracted features directory                        | `./outputs/features` |
| `paths.tokenized`          | Tokenized data directory                            | `./outputs/tokenized` |
| `paths.data`               | Raw EHR data directory                              | `./example_data/example_data_w_labs` |
| `split_ratios.test`        | Percentage of data for testing                      | `0.1` |
| `trainer_args.batch_size`  | Training batch size                                 | `32` |
| `trainer_args.epochs`      | Number of training epochs                           | `5` |
| `optimizer.lr`             | Learning rate                                       | `5e-4` |