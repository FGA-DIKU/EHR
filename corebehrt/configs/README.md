# CoreBEHRT Configuration Files Overview 
This repository contains configuration files for processing Electronic Health Record (EHR) data using CoreBEHRT, providing an overview of multiple configuration files used in different stages of data processing and modeling.  

### **Create Data (`create_data.yaml`)**  
This step **loads and processes raw EHR data**, extracts key clinical concepts, tokenizes records, and prepares structured inputs for modeling.  

#### **Key Functions:**  
- Loads and processes **raw EHR data**, extracting **diagnoses, medications, procedures, and lab tests**.  
- Defines **data paths** for raw data, tokenized sequences, and extracted features.  
- Tokenizes **patient records** into structured sequences for modeling.  
- Extracts **background variables** and sets a **reference timestamp** .  
- Configures **value processing**, including:  
  - **Binning** values into categories.  
  - **Normalization** for feature scaling.  
  - **Handling missing values**.  
- Splits the dataset into: **pretraining, finetuning, testing**  
---

### **Hyperparameters for `create data stage`**  

| **Parameter**        | **Description**                                            | **Value** |
|----------------------|------------------------------------------------------------|----------|
| `logging.level`      | Log verbosity level                                        | `INFO`   |
| `logging.path`       | Path to save logs                                          | `./outputs/logs` |
| `paths.data`         | Raw EHR data directory                                    | `./example_data/example_data_w_labs` |
| `paths.tokenized`    | Tokenized data directory                                  | `./outputs/tokenized` |
| `paths.features`     | Extracted features directory                              | `./outputs/features` |
| `loader.concept_types` | Types of concepts to extract (`diagnoses`, `medications`, `procedures`, `lab tests`) | `["diagnose", "medication", "labtest"]` |
| `loader.include_values` | Values to include                                      | `["labtest"]` |
| `features.background_vars` | Background demographic variables                     | `["GENDER"]` |
| `features.origin_point` | Reference timestamp                                    | `2020-01-26` |
| `values.value_type`  | Type of value binning                                     | `binned` |
| `values.normalize.func` | Normalization function                                 | `min_max_normalize_results` |
| `values.normalize.kwargs.min_count` | Minimum count for normalization          | `3` |
| `excluder.min_age`   | Minimum age for inclusion                                | `-1` |
| `excluder.max_age`   | Maximum age for inclusion                                | `120` |
| `split_ratios.pretrain` | Percentage of data for pretraining                    | `0.72` |
| `split_ratios.finetune` | Percentage of data for finetuning                     | `0.18` |
| `split_ratios.test`  | Percentage of data for testing                           | `0.1` |

---
### Pretrain (`pretrain.yaml`)  
- Trains a **transformer-based model** on **EHR sequences** using **masked language modeling (MLM)**.  
- Loads **tokenized patient records** and **structured features** as inputs.  
- Applies **80% masking** and **10% token replacement** during MLM training.  
- Uses a **truncation length of 20** and filters out sequences **shorter than 2 tokens**.  
- Splits data into **80% training and 20% validation**.  
- Trains for **5 epochs** with a **batch size of 32** (effective batch size: **64**).  
- Optimizes using **Adam with LR = 5e-4**, **gradient clipping (1.0)**, and **linear warmup for 2 epochs**.  
- Saves pretrained models to `./outputs/pretraining/`.  
- Monitors performance using **top-1/top-10 precision** and **MLM loss**.  

### **Pretrain (`pretrain`)**  
This step **trains a transformer-based model** on **EHR sequences** using **masked language modeling (MLM)** to learn meaningful patient data representations.  

#### **Key Functions:**  
- Loads **tokenized patient records** and **structured features** as inputs.  
- Applies **80% masking** and **10% token replacement** during MLM training.  
- Uses a **truncation length of 20** and filters out sequences **shorter than 2 tokens**.  
- Splits data into:  
  - **training**  
  - **validation**  
- Trains for **5 epochs** with:  
  - **Batch size of 32** (effective batch size: **64**)  
  - **Gradient clipping** to stabilize training  
- Optimizes using **Adam optimizer** 
- Saves pretrained models to `./outputs/pretraining/`.  
- Monitors performance using **top-1/top-10 precision** and **MLM loss**.  

---

### **Hyperparameters for `Pretrain`**  

| **Parameter**        | **Description**                                          | **Value** |
|----------------------|----------------------------------------------------------|----------|
| `logging.level`      | Log verbosity level                                      | `INFO`   |
| `logging.path`       | Path to save logs                                        | `./outputs/logs` |
| `paths.features`     | Extracted features directory                            | `./outputs/features` |
| `paths.tokenized`    | Tokenized data directory                                | `./outputs/tokenized` |
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

This version ensures **clarity, structure, and easy reference** while keeping the documentation **concise and well-organized**. 🚀 Let me know if you need any refinements! 😊


### Define Outcomes (`outcome`)  
- Extracts and processes **clinical outcome labels** from EHR records.  
- Loads **diagnosis-related concepts** with a batch size of **10,000**.  
- Filters and structures outcome data from `./example_data/example_data_w_labs`.  
- Defines **TEST_OUTCOME**:  
  - Matches concepts **157, 169** while **excluding 157141000119108**.  
  - Uses **case-sensitive matching** and checks if values **contain** specified concepts.  
- Defines **TEST_CENSOR**:  
  - Matches concept **169** using **case-insensitive "startswith" filtering**.  
- Saves processed outcome labels to `./outputs/outcomes/`.  


### **Select Cohort (`select_cohort`)**  
This configuration selects a **subset of patients** based on predefined criteria for further analysis.
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
  - **80% training**, **10% validation**, and **10% testing**.  
- Saves the final **cohort data** to `./outputs/cohort/`.  

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

This step **fine-tunes the model on clinical outcomes** and then **evaluates its predictive performance** before deployment. 🚀  



## Notes  
- Modify **hyperparameters, dataset paths, and selection criteria** based on your needs.
- Ensure all configuration files specially paths are properly set before running the pipeline.  
- For detailed configurations, check the respective YAML/JSON files.  

# Common Items
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

