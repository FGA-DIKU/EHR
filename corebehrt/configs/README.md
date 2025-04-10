# CoreBEHRT Configuration Files Overview
This repository contains configuration files for processing Electronic Health Record (EHR) data using CoreBEHRT. Below is an overview of the key configuration files used in different stages of data processing and modeling.

In the following sections, each configuration file is explained in detail.

# Create Data

Below is a detailed breakdown of all configurable parameters used during the `create data` stage in CoreBEHRT.



#  Configuration Hyperparameters for the `create data` stage

| **Category** | **Parameter**                                  | **Default**               | **Possible Values**                                 | **Required?**      | **Description**                                                  |
|--------------|------------------------------------------------|---------------------------|-----------------------------------------------------|--------------------|------------------------------------------------------------------|
| `logging`    | `level`                                        | `INFO`                    | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`     | Yes                | Logging level that controls verbosity of output logs.            |
|              | `path`                                         | `./outputs/logs`          | any valid path                                      | Yes                | Directory where log files will be stored.                        |
| `paths`      | `data`                                         | `./example_data/...`      | any valid path                                      | Yes                | Path to the raw EHR input data.                                  |
|              | `tokenized`                                    | `./outputs/tokenized`     | any valid path                                      | Yes                | Directory to store tokenized patient records.                    |
|              | `features`                                     | `./outputs/features`      | any valid path                                      | Yes                | Directory to save extracted features.                            |
|              | `code_mapping`                                 | *(not set)*               | any valid path to `.pt` file                        | No                 | Optional path to save/load code mapping.                         |
|              | `vocabulary`                                   | *(not set)*               | any valid path                                      | No                 | Optional path to the vocabulary folder.                          |
| `features`   | `exclude_regex`                                | `^(?:LAB).*`              | any valid regex                                     | No                 | Regex pattern to exclude specific feature types.                 |
|              | `values.value_creator_kwargs.num_bins`         | `100`                     | any positive integer                                | If labs            | Number of bins for discretizing numeric feature values.          |
| `tokenizer`  | `sep_tokens`                                   | `true`                    | `true`, `false`                                     | No                 | Whether to include separator tokens between events.              |
|              | `cls_token`                                    | `true`                    | `true`, `false`                                     | No                 | Whether to include a classification token at the beginning.      |
| `excluder`   | `min_age`                                      | `-1`                      | any integer                                         | No                 | Minimum age for patients to be included.                         |
|              | `max_age`                                      | `120`                     | any integer                                         | No                 | Maximum age for patients to be included.                         |


---

## 📌 **Next Steps**
Now we need to refine **default values and possible values** for each parameter. Let me know which parameters need updates or if you'd like to add more details! 🚀


(For shared parameters, refer to [Common Hyperparameters](#common-hyperparameters-shared-across-all-stages))_

---
---

### **Prepare Training Data**

This step **converts tokenized sequences and structured data** into pretraining- or fine-tuning-ready formats.

#### Key Functions:
- Reads tokenized sequences and extracted features.
- Applies dataset filtering based on minimum length and configuration.
- Converts inputs to the required format for model training.
- Creates training/validation splits according to specified ratios.
- Stores the prepared dataset to the appropriate output directory.

#### Usage Examples:
```bash
# For pretraining
(.venv) python -m corebehrt.main.prepare_training_data --config_path corebehrt/configs/prepare_pretrain.yaml

# For fine-tuning
(.venv) python -m corebehrt.main.prepare_training_data --config_path corebehrt/configs/prepare_finetune.yaml

---------

### **Pretrain**
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


trainer_args:
  sampler: null
WeightedRandomSampler
RandomSampler
SequentialSampler
SubsetRandomSampler
BatchSampler




# Gradient Clipping: Key Values & Effects

`gradient_clip.clip_value` controls how much gradients are limited to prevent exploding gradients.

## 🔹 Recommended Values

| **Value**   | **Effect** |
|------------|-----------|
| **`None` / `False`** | No clipping, risk of exploding gradients. |
| **`0.1 - 0.5`** | Strong clipping, slows learning. |
| **`1.0` (default)** | Balanced, prevents instability. |
| **`5.0`** | More flexibility, still prevents explosion. |
| **`10.0+`** | Almost no clipping, useful if explosion isn’t an issue. |

## 🔹 Best Practices
✅ Use **`1.0 - 5.0`** for stability.  
✅ Use **`0.5`** if gradients are noisy.  
✅ Set to **`None`** if no explosion issues.  


| **Category**     | **Parameter**                        | **Value** |
|-----------------|-------------------------------------|----------|
| **Data**        | `dataset.select_ratio`                     | `1.0` |
|                 | `dataset.masking_ratio`                    | `0.8` |
|                 | `dataset.replace_ratio`                    | `0.1` |
|                 | `dataset.ignore_special_tokens`            | `true` |
|                 | `truncation_len`                   | `20` |
|                 | `val_ratio`                        | `0.2` |
|                 | `min_len`                          | `2` |
| **training arguments**| `batch_size`                       | `32` |
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

(For shared parameters, refer to [Common Hyperparameters](#common-hyperparameters-shared-across-all-stages))_


---
### **Define Outcomes**
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

### Hyperparameters for `outcome.yaml`
| **Category**    | **Parameter**         | **Value** |
|---------------|---------------------|----------|
| **Paths**    | outcomes            | `./outputs/outcomes` |
| **Loader**   | concepts            | `["diagnose"]` |
|             | batchsize           | `Optimized batch size` |
|             | chunksize           | `Optimized chunk size` |
| **TEST_OUTCOME** | type             | `["CONCEPT"]` |
|             | match              | `Predefined set` |
|             | exclude            | `Predefined exclusions` |
|             | match_how          | `contains` |
|             | case_sensitive     | `true` |
| **TEST_CENSOR**  | type             | `["CONCEPT"]` |
|             | match              | `Predefined set` |
|             | match_how          | `startswith` |
|             | case_sensitive     | `false` |

## Notes  
- The batch size and chunk size are **optimized for efficiency** without compromising accuracy.  
- `TEST_OUTCOME` and `TEST_CENSOR` use **different matching rules** to ensure precise classification.  
- Processed outcomes are **saved to `./outputs/outcomes/`** for further analysis.  

(For shared parameters, refer to [Common Hyperparameters](#common-hyperparameters-shared-across-all-stages))_


---

### **select cohort**  

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

### **Hyperparameters for `select_cohort.yaml`**  
| **Category**       | **Parameter**                     | **Value** |
|-------------------|---------------------------------|----------------------------------------------|
| **Paths**        | `patients_info`                | `./example_data/example_data_w_labs/patients_info.csv` |
|                 | `initial_pids`                 | `./outputs/tokenized/pids_finetune.pt` |
|                 | `exposure`                     | `./outputs/outcomes/TEST_CENSOR.csv` |
|                 | `outcome`                      | `./outputs/outcomes/TEST_OUTCOME.csv` |
|                 | `cohort`                       | `./outputs/cohort/` |
| **Selection**    | `exclude_prior_outcomes`      | `true` |
|                 | `exposed_only`                 | `false` |
|                 | `age.min_years`                | `Configured limit` |
|                 | `age.max_years`                | `Configured limit` |
|                 | `categories.GENDER.include`    | `Predefined category` |
| **Index Date**   | `mode`                         | `relative` |
|                 | `absolute.year`                | `Configured date` |
|                 | `absolute.month`               | `Configured date` |
|                 | `absolute.day`                 | `Configured date` |
|                 | `relative.n_hours_from_exposure` | `Configured offset` |
| **Split Ratios** | `train`                        | `Configured percentage` |
|                 | `val`                          | `Configured percentage` |
|                 | `test`                         | `Configured percentage` |

(For shared parameters, refer to [Common Hyperparameters](#common-hyperparameters-shared-across-all-stages))_

---
### **Fine-Tune & Evaluate**  

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

## **Hyperparameters for `fine_tune.yaml` & `finetune_evaluate.yaml`**  


| **Category**         | **Parameter**                        | **Value** |
|---------------------|------------------------------------|----------------------------------------------|
| **Paths**          | `pretrain_model`                   | `./outputs/pretraining/` |
|                   | `outcome`                          | `./outputs/outcomes/TEST_OUTCOME.csv` |
|                   | `model`                            | `./outputs/finetuning/` |
| **Model**         | `cls._target_`                     | `ehr2vec.model.heads.ClassifierGRU` |
|                   | `cls.bidirectional`                | `true` |
| **Cross-Validation** | `cv_splits`                     | `Configured value` |
| **Data**          | `val_split`                        | `Configured percentage` |
|                   | `truncation_len`                   | `Configured limit` |
|                   | `min_len`                          | `Configured limit` |
| **Outcome**       | `n_hours_censoring`                | `Configured offset` |
|                   | `n_hours_start_follow_up`          | `Configured time` |
|                   | `n_hours_end_follow_up`            | `Configured time` |
| **Training Arguments** | `val_batch_size`              | `Configured size` |
|                   | `effective_batch_size`             | `Configured size` |
|                   | `gradient_clip.clip_value`         | `Configured value` |
|                   | `shuffle`                          | `true` |
|                   | `early_stopping`                   | `Configured patience` |
|                   | `stopping_criterion`               | `roc_auc` |
| **Optimizer**     | `eps`                              | `Configured value` |
| **Scheduler**     | `_target_`                         | `transformers.get_linear_schedule_with_warmup` |
|                   | `num_warmup_steps`                 | `Configured steps` |
|                   | `num_training_steps`               | `Configured steps` |
| **Metrics**       | `accuracy._target_`                | `corebehrt.modules.monitoring.metrics.Accuracy` |
|                   | `accuracy.threshold`              | `Configured threshold` |
|                   | `roc_auc._target_`                | `corebehrt.modules.monitoring.metrics.ROC_AUC` |

|                   | `pr_auc._target_`                 | `corebehrt.modules.monitoring.metrics.PR_AUC` |
|                   | `precision._target_`              | `corebehrt.modules.monitoring.metrics.Precision` |
|                   | `recall._target_`                 | `corebehrt.modules.monitoring.metrics.Recall` |
|                   | `mean_probability._target_`       | `corebehrt.modules.monitoring.metrics.Mean_Probability` |
|                   | `percentage_positives._target_`   | `corebehrt.modules.monitoring.metrics.Percentage_Positives` |
|                   | `true_positives._target_`         | `corebehrt.modules.monitoring.metrics.True_Positives` |
|                   | `true_negatives._target_`         | `corebehrt.modules.monitoring.metrics.True_Negatives` |
|                   | `false_positives._target_`        | `corebehrt.modules.monitoring.metrics.False_Positives` |
|                   | `false_negatives._target_`        | `corebehrt.modules.monitoring.metrics.False_Negatives` |

(For shared parameters, refer to [Common Hyperparameters](#common-hyperparameters-shared-across-all-stages))_

---
## Common Items
### **Common hyperparameters (Shared Across All Stages)**  

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