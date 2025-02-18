# CoreBEHRT Configuration Files Overview 
This repository contains configuration files for processing **Electronic Health Record (EHR) data** using CoreBEHRT.

This document provides an overview of multiple configuration files used in different stages of data processing and modeling. Each configuration file serves a specific purpose, such as data creation, pretraining, finetuning, evaluation, and cohort selection. Below is a summary of each configuration file.

## Pipeline Overview  
The configuration files define different stages of data processing and modeling in the CoreBEHRT pipeline. These stages include:

### Create Data (`create_data`)  
- Loads and processes **raw EHR data**, extracting key clinical concepts (**diagnoses, medications, procedures ,lab tests**).  
- Defines **data paths** for raw, tokenized, and feature-extracted data.  
- Tokenizes patient records into structured sequences for modeling.  
- Extracts **background variables** (e.g., `GENDER`) and sets a **reference timestamp** (`2020-01-26`).  
- Configures **value processing**, including **binning, normalization**, and **handling missing values**.  
- Splits the dataset into **pretraining (72%), finetuning (18%), and test (10%)** subsets.  


### Pretrain (`pretrain`)  
- Trains a **transformer-based model** on **EHR sequences** using **masked language modeling (MLM)**.  
- Loads **tokenized patient records** and **structured features** as inputs.  
- Applies **80% masking** and **10% token replacement** during MLM training.  
- Uses a **truncation length of 20** and filters out sequences **shorter than 2 tokens**.  
- Splits data into **80% training and 20% validation**.  
- Trains for **5 epochs** with a **batch size of 32** (effective batch size: **64**).  
- Optimizes using **Adam with LR = 5e-4**, **gradient clipping (1.0)**, and **linear warmup for 2 epochs**.  
- Saves pretrained models to `./outputs/pretraining/`.  
- Monitors performance using **top-1/top-10 precision** and **MLM loss**.  


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

