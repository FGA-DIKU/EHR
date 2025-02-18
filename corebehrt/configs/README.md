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
- Specifies **clinical outcome labels** from EHR records.  
- Defines **inclusion/exclusion criteria** for patient events.  

###  Select Cohort (`select_cohort`)  
- Filters patients based on **age, gender, diagnoses, and exposures**.  
- Creates a study group for further analysis.  

###  Fine-Tune & Evaluate (`fine_tune` & `finetune_evaluate`)  
- Fine-tunes the pretrained model for **predicting clinical outcomes**.  
- Evaluates performance using **accuracy, precision, recall, and AUC scores**.  

Each section below summarizes the key components of these configurations.


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

