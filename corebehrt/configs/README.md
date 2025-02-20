# CoreBEHRT Configuration Files Overview 
This repository contains configuration files for processing Electronic Health Record (EHR) data using CoreBEHRT, providing an overview of multiple configuration files used in different stages of data processing and modeling.  

### **Create Data (`create_data.yaml`)**  
This step **loads and processes raw EHR data**, extracts key clinical concepts, tokenizes records, and prepares structured inputs for modeling.  

#### **Key Functions:**  
- Loads and processes **raw EHR data**, extracting **diagnoses, medications, procedures and lab tests**.  
- Defines **data paths** for raw data, tokenized sequences, and extracted features.  
- Tokenizes **patient records** into structured sequences for modeling.  
- Extracts **background variables** and sets a **reference timestamp** .  
- Configures **value processing**, including:  
  - **Binning** values into categories.  
  - **Normalization** for feature scaling.  
  - **Handling missing values**.  
- Splits the dataset into: **pretrain, finetune and test**  

### **Hyperparameters for the `create data` stage**  

_(For shared parameters, refer to [Common Hyperparameters](#common-hyperparameters))_  

| **Parameter**                     | **Description**                                           | **Value** |
|-----------------------------------|-----------------------------------------------------------|----------|
| `loader.concept_types`            | Types of concepts to extract  | `["diagnose", "medication","procedure",  "labtest"]` |
| `loader.include_values`           | Values to include                                       | `["labtest"]` |
| `features.background_vars`        | Background demographic variables                        | `["GENDER"]` |
| `features.origin_point`           | Reference timestamp                                     | `2020-01-26` |
| `values.value_type`               | Type of value binning                                  | `binned` |
| `values.normalize.func`           | Normalization function                                | `min_max_normalize_results` |
| `values.normalize.kwargs.min_count` | Minimum count for normalization                     | `3` |
| `excluder.min_age`                | Minimum age for inclusion                            | `-1` |
| `excluder.max_age`                | Maximum age for inclusion                            | `120` |
| `split_ratios.pretrain`           | Percentage of data for pretraining                   | `0.72` |
| `split_ratios.finetune`           | Percentage of data for finetuning                    | `0.18` |

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

### **Hyperparameters for `Pretrain`**  stage

| **Parameter**        | **Description**                                          | **Value** |
|----------------------|----------------------------------------------------------|----------|
| `data.dataset.select_ratio` | Percentage of dataset to use                     | `1.0` |
| `data.dataset.masking_ratio` | Percentage of tokens masked                      | `0.8` |
| `data.dataset.replace_ratio` | Percentage of tokens replaced                    | `0.1` |
| `data.dataset.ignore_special_tokens` | Ignore special tokens in masking         | `true` |
| `data.truncation_len` | Maximum sequence length                                | `20` |
| `data.val_ratio`     | Validation set percentage                               | `0.2` |
| `data.min_len`      | Minimum sequence length                                 | `2` |
| `trainer_args.batch_size` | Training batch size                                 | `32` |
| `trainer_args.effective_batch_size` | Effective batch size                     | `64` |
| `trainer_args.epochs` | Number of training epochs                              | `5` |
| `trainer_args.gradient_clip.clip_value` | Gradient clipping value               | `1.0` |
| `trainer_args.shuffle` | Shuffle training data                                 | `true` |
| `optimizer.lr`       | Learning rate                                           | `5e-4` |
| `optimizer.eps`      | Epsilon value for Adam optimizer                        | `1e-6` |
| `scheduler._target_` | Learning rate scheduler                                | `transformers.get_linear_schedule_with_warmup` |
| `scheduler.num_warmup_epochs` | Number of warmup epochs                        | `2` |
| `scheduler.num_training_epochs` | Total training epochs                        | `3` |
| `metrics.top1._target_` | Top-1 precision metric                               | `corebehrt.modules.monitoring.metrics.PrecisionAtK` |
| `metrics.top1.topk` | Top-k value for top-1 precision                         | `1` |
| `metrics.top10._target_` | Top-10 precision metric                             | `corebehrt.modules.monitoring.metrics.PrecisionAtK` |
| `metrics.top10.topk` | Top-k value for top-10 precision                        | `10` |
| `metrics.mlm_loss._target_` | MLM loss function                                | `corebehrt.modules.monitoring.metrics.LossAccessor` |
| `metrics.mlm_loss.loss_name` | Name of the loss function                       | `loss` |

---
# Define Outcomes (`outcome.yaml`)
This step **extracts and processes clinical outcome labels** from EHR records to support downstream modeling. The process is designed for efficiency, handling large volumes of diagnosis-related concepts while ensuring accurate filtering and classification.

## Key Functions

### Efficient Data Loading  
- Loads diagnosis-related concepts in **optimized batches** to enhance processing speed.  
- Reads data in **manageable chunks** for efficient handling.  

### Outcome Labeling  
- **Defines `TEST_OUTCOME`**:  
  - Identifies specific clinical concepts but **excludes certain values** based on predefined criteria.  
  - Uses **case-sensitive matching** with a **"contains" filter** for concept identification.  
- **Defines `TEST_CENSOR`**:  
  - Matches relevant concepts using **case-insensitive "startswith" filtering**.  

### Output Storage  
- Saves processed **outcome labels** to `./outputs/outcomes/`.  

## Hyperparameters for `outcome`

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
### **Select Cohort (`select_cohort`)**  
This configuration **selects a subset of patients** based on predefined criteria for further analysis.  

#### **Key Functions:**  
- Loads **patient information** from `patients_info.csv`.  
- Filters patients based on:  
  - **Age range**: Includes only patients between **18 - 120 years**.  
  - **Gender**: Includes only **male (`M`)** patients.  
  - **Exposure status**: Uses `TEST_CENSOR.csv`, if provided.  
  - **Outcome history**: Excludes patients who had the outcome **before the index date**.  
- Defines the **index date** as:  
  - **Absolute date**: `2015-01-26`.  
  - **Relative to exposure**: `24 hours before first exposure`.  
- Splits the selected cohort into:  
  - **training, validation, and testing**.  
- Saves the final **cohort data** to `./outputs/cohort/`.  

---

### **Hyperparameters for `select_cohort`**  

| **Parameter**              | **Description**                                            | **Value** |
|----------------------------|------------------------------------------------------------|----------|
| `logging.level`            | Log verbosity level                                        | `INFO`   |
| `logging.path`             | Path to save logs                                          | `./outputs/logs` |
| `paths.patients_info`      | Patient information file                                   | `./example_data/example_data_w_labs/patients_info.csv` |
| `paths.initial_pids`       | Initial patient IDs (optional)                            | `./outputs/tokenized/pids_finetune.pt` |
| `paths.exposure`           | Exposure file (optional)                                  | `./outputs/outcomes/TEST_CENSOR.csv` |
| `paths.outcome`            | Outcome file                                             | `./outputs/outcomes/TEST_OUTCOME.csv` |
| `paths.cohort`             | Output directory for cohort data                         | `./outputs/cohort/` |
| `selection.exclude_prior_outcomes` | Exclude patients with prior outcomes             | `true` |
| `selection.exposed_only`   | Include only exposed patients                            | `false` |
| `selection.age.min_years`  | Minimum age for inclusion                                | `18` |
| `selection.age.max_years`  | Maximum age for inclusion                                | `120` |
| `selection.categories.GENDER.include` | Gender selection                              | `["M"]` |
| `index_date.mode`          | Index date mode (absolute/relative)                     | `relative` |
| `index_date.absolute.year` | Absolute index year                                     | `2015` |
| `index_date.absolute.month`| Absolute index month                                    | `1` |
| `index_date.absolute.day`  | Absolute index day                                      | `26` |
| `index_date.relative.n_hours_from_exposure` | Relative index time before exposure    | `-24` |
| `split_ratios.train`       | Percentage of data for training                         | `0.8` |
| `split_ratios.val`         | Percentage of data for validation                       | `0.1` |
| `split_ratios.test`        | Percentage of data for testing                          | `0.1` |

---

###  Fine-Tune & Evaluate (`fine_tune` & `finetune_evaluate`)  
- Fine-tunes the pretrained model for **predicting clinical outcomes**.  
- Evaluates performance using **accuracy, precision, recall, and AUC scores**.  

### **Fine-Tune & Evaluate (`fine_tune` & `finetune_evaluate`)**  
This phase **fine-tunes** the pretrained model on specific clinical outcomes and **evaluates** its performance using various metrics.

#### **🔹 Fine-Tuning (`fine_tune`)**
- Loads **pretrained model** from `./outputs/pretraining/`.  
- Uses **tokenized data, extracted features, and cohort selection** for training.  
- **Trains a classifier (`ClassifierGRU`)** in a **bidirectional** mode.  
- Converts **outcome labels to binary values** based on their presence in a follow-up window.  
- Uses a **truncation length of 30** and removes sequences **shorter than 2 tokens**.  
- Splits the data into:  
  - **10% validation**  
  - **10% test set**  
- Trains for **3 epochs** with:  
  - **Batch size of 8**  
  - **Validation batch size of 16**  
  - **Gradient clipping (1.0)**  
  - **Early stopping after 20 epochs**  
- Uses **learning rate `5e-4`** with **warmup steps (10)** and **total training steps (100)**.  
- Saves the **fine-tuned model** to `./outputs/finetuning/`.  

#### **🔹 Evaluation (`finetune_evaluate`)**
- Loads the **fine-tuned model** from `../outputs/pretraining/test/finetune_TEST_OUTCOME_censored_5_days_post_TEST_OUTCOME_test`.  
- Runs evaluation using **a test dataset** located in the same directory.  
- Computes various **performance metrics**, including:  
  - **Accuracy** (Threshold: `0.6`)  
  - **Balanced Accuracy**  
  - **Precision & Recall**  
  - **ROC-AUC & PR-AUC**  
  - **True/False Positives & Negatives**  
  - **Mean Probability & Percentage of Positives**  



### **Fine-Tune & Evaluate (`fine_tune` & `finetune_evaluate`)**  
This phase **fine-tunes** the pretrained model on specific clinical outcomes and **evaluates** its performance using various metrics.

#### **🔹 Fine-Tuning (`fine_tune`)**
- Loads **pretrained model** from `./outputs/pretraining/`.  
- Uses **tokenized data, extracted features, and cohort selection** for training.  
- **Trains a classifier (`ClassifierGRU`)** in a **bidirectional** mode.  
- Converts **outcome labels to binary values** based on their presence in a follow-up window.  
- Uses a **truncation length of 30** and removes sequences **shorter than 2 tokens**.  
- Splits the data into:  
  - **10% validation**  
  - **10% test set**  
- Trains for **3 epochs** with:  
  - **Batch size of 8**  
  - **Validation batch size of 16**  
  - **Gradient clipping (1.0)**  
  - **Early stopping after 20 epochs**  
- Uses **learning rate `5e-4`** with **warmup steps (10)** and **total training steps (100)**.  
- Saves the **fine-tuned model** to `./outputs/finetuning/`.  

#### **🔹 Evaluation (`finetune_evaluate`)**
- Loads the **fine-tuned model** from `../outputs/pretraining/test/finetune_TEST_OUTCOME_censored_5_days_post_TEST_OUTCOME_test`.  
- Runs evaluation using **a test dataset** located in the same directory.  
- Computes various **performance metrics**, including:  
  - **Accuracy** (Threshold: `0.6`)  
  - **Balanced Accuracy**  
  - **Precision & Recall**  
  - **ROC-AUC & PR-AUC**  
  - **True/False Positives & Negatives**  
  - **Mean Probability & Percentage of Positives**  

This step **fine-tunes the model on clinical outcomes** and then **evaluates its predictive performance** before deployment. 

---

###  **Hyperparameters for `fine_tune` & `finetune_evaluate`**  

| **Parameter**                        | **Description**                                          | **Value** |
|--------------------------------------|----------------------------------------------------------|----------|
| `logging.level`                      | Log verbosity level                                      | `INFO`   |
| `logging.path`                       | Path to save logs                                       | `./outputs/logs` |
| `paths.features`                     | Extracted features directory                            | `./outputs/features` |
| `paths.tokenized`                    | Tokenized data directory                                | `./outputs/tokenized` |
| `paths.cohort`                       | Cohort selection directory                             | `./outputs/cohort` |
| `paths.pretrain_model`               | Path to pretrained model                               | `./outputs/pretraining/` |
| `paths.outcome`                      | Outcome file location                                  | `./outputs/outcomes/TEST_OUTCOME.csv` |
| `paths.model`                        | Path to save fine-tuned model                         | `./outputs/finetuning/` |
| `model.cls._target_`                 | Model architecture (ClassifierGRU)                    | `ehr2vec.model.heads.ClassifierGRU` |
| `model.cls.bidirectional`            | Whether the GRU model is bidirectional                | `true` |
| `cv_splits`                          | Number of cross-validation splits                     | `2` |
| `data.val_split`                     | Validation set percentage                             | `0.1` |
| `data.test_split`                    | Test set percentage                                   | `0.1` |
| `data.truncation_len`                | Maximum sequence length                               | `30` |
| `data.min_len`                       | Minimum sequence length                               | `2` |
| `outcome.n_hours_censoring`          | Hours to censor after index date                     | `-10` |
| `outcome.n_hours_start_follow_up`    | Start of follow-up period                            | `1` |
| `outcome.n_hours_end_follow_up`      | End of follow-up period                              | `null` |
| `trainer_args.batch_size`            | Training batch size                                  | `8` |
| `trainer_args.val_batch_size`        | Validation batch size                                | `16` |
| `trainer_args.effective_batch_size`  | Effective batch size                                 | `16` |
| `trainer_args.epochs`                | Number of training epochs                           | `3` |
| `trainer_args.gradient_clip.clip_value` | Gradient clipping value                          | `1.0` |
| `trainer_args.shuffle`               | Shuffle training data                               | `true` |
| `trainer_args.early_stopping`        | Early stopping patience                             | `20` |
| `trainer_args.stopping_criterion`    | Stopping criterion                                  | `roc_auc` |
| `optimizer.lr`                       | Learning rate                                       | `5e-4` |
| `optimizer.eps`                      | Epsilon value for Adam optimizer                    | `1e-6` |
| `scheduler._target_`                 | Learning rate scheduler                            | `transformers.get_linear_schedule_with_warmup` |
| `scheduler.num_warmup_steps`         | Number of warmup steps                              | `10` |
| `scheduler.num_training_steps`       | Total number of training steps                      | `100` |
| `metrics.accuracy._target_`          | Accuracy metric                                    | `corebehrt.modules.monitoring.metrics.Accuracy` |
| `metrics.accuracy.threshold`         | Accuracy threshold                                 | `0.6` |
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




## Notes  
- Modify **hyperparameters, dataset paths, and selection criteria** based on your needs.
- Ensure all configuration files specially paths are properly set before running the pipeline.  
- For detailed configurations, check the respective YAML/JSON files.  

# Common Items
## **Common Hyperparameters (Shared Across All Stages)**  

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

## Logging Configuration

### Purpose
This section defines how logging should be handled in the model, ensuring that execution details are recorded for debugging/ monitoring.

### Level
Defines the logging severity level.

- **Options**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Current setting**: `INFO` (Logs general system messages)

## Path
### Directory Configuration for Input and Output Data

This document defines the directories used for input and output data in various processes.

| Parameter   | Description                          | Used In                        | Order of Usage                        |
|------------|--------------------------------------|--------------------------------|--------------------------------------|
| **data**   | Location of input data              | `create_data`, `outcome`       | Raw input data                      |
| **tokenized** | Directory for tokenized data    | `create_data`, `pretrain`      | Data preprocessing (tokenization)   |
| **features**  | Directory for extracted features  | `create_data`, `pretrain`, `outcome` | Feature extraction (used in multiple stages) |
| **model**     | Output directory for saved models | `pretrain`, `fine_tune`        | Model training begins (pretraining, fine-tune) |
| **cohort**    | Directory for patient cohort data | `fine_tune`, `select_cohort`   | Selected patient data for fine-tuning |
| **outcomes**  | Directory for outcome data       | `outcome`, `fine_tune`         | Final results after training/fine-tuning |


## Data Processing & Loading  

This section defines the parameters for data ingestion, transformation, and processing.  

| Parameter                      | Description                                                        | Used In             | Order of Execution                           |
|---------------------------------|--------------------------------------------------------------------|---------------------|----------------------------------------------|
| **Concept Types**               | Specifies the medical concepts to be included (e.g., `diagnose`, `medication`, `labtest`). | `create_data`       | Define relevant medical concepts.           |
| **Normalization**               | Standardizes numerical values to ensure consistency across datasets. | `create_data`       | Apply data normalization techniques.        |
| **Batch Size**                  | Determines the volume of data processed in a single iteration. | `outcome`           | Configure batch size for processing.        |
| **Truncation Length**           | Sets a maximum limit on sequence lengths to maintain uniformity. | `pretrain`, `fine_tune` | Enforce sequence length constraints.  |
| **Masking & Replacement Ratios** | Defines the proportion of data to be masked or replaced during preprocessing. | `pretrain`          | Apply masking and replacement strategies.   |

        |

