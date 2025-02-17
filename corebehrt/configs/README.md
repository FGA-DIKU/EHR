# CoreBEHRT Configuration Files
This repository contains configuration files for processing **Electronic Health Record (EHR) data** using CoreBEHRT.

## Pipeline Overview
The CoreBEHRT pipeline follows these key steps:

### 1- Create Data  
Processes raw **EHR data**, extracts relevant clinical concepts (**diagnoses, medications, lab tests**), tokenizes records, and generates structured features.

### 2️- Pretrain  
Trains a transformer-based model on EHR sequences using **masked language modeling (MLM)** to learn patient data representations.

### 3️- Create Outcome  
Defines and extracts **clinical outcome labels** from EHR records, specifying **inclusion/exclusion criteria** for patient events.



### 4️- Select Cohort  
Filters patients based on **age, gender, prior diagnoses, exposure, and other clinical criteria** to create a study group.

### 5️- Fine-Tune & Evaluate  
Fine-tunes the pretrained model for **predicting clinical outcomes** and evaluates performance using metrics like **accuracy, precision, recall, and AUC scores**.

## Notes  
- Modify **hyperparameters, dataset paths, and selection criteria** based on your needs.
- Ensure all configuration files specially paths are properly set before running the pipeline.  
- For detailed configurations, check the respective YAML/JSON files.  

# Common Items
## Logging Configuration

## Purpose
This section defines how logging should be handled in the model, ensuring that execution details are recorded for debugging/ monitoring.

### Level
Defines the logging severity level.

- **Options**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Current setting**: `INFO` (Logs general system messages)

### Path
# Directory Configuration for Input and Output Data

This document defines the directories used for input and output data in various processes.

| Parameter   | Description                          | Used In                        | Default Value                        |
|------------|--------------------------------------|--------------------------------|--------------------------------------|
| **data**   | Location of input data              | `create_data`, `outcome`       | `./example_data/example_data_w_labs` |
| **tokenized** | Directory for tokenized data    | `create_data`, `pretrain`      | `./outputs/tokenized`               |
| **features**  | Directory for extracted features  | `create_data`, `pretrain`, `outcome` | `./outputs/features`          |
| **outcomes**  | Directory for outcome data       | `outcome`, `fine_tune`         | `./outputs/outcomes`                |
| **cohort**    | Directory for patient cohort data | `fine_tune`, `select_cohort`   | `./outputs/cohort`                  |
| **model**     | Output directory for saved models | `pretrain`                     | `./outputs/pretraining`             |



## create_data

## create_outcomes