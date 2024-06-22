# EHR

## 1. Create Data
The pretraining process is handled by the [`main_create_data.py`](corebehrt/main_create_data.py) script.
### [`FeatureCreator`](corebehrt/classes/features.py)

From the **raw data**
PID|CONCEPT|ADMISSION_ID|TIMESTAMP|...
---|-------|------------|---------|---

and **patient data**

PID|GENDER|birthcolumn|deathcolumn|...
---|------|-----------|---------|---

we create: 

PID|concept|abspos|segment|age|...
---|-------|------|-----|---|---

and include the following:
- background 
- death event

### [`Excluder`](corebehrt/classes/excluder.py)
- incorrect ages
- nans
- normalize segments after exclusion
- short sequences

Results are saved in a table.

### [`Batches`](corebehrt/data/batch.py)
Takes care of loading in batches (currently from sequence files)  
Splitting data into pretrain/finetune/test.  
Possible to assign pids to different sets.

### [`EHRTokenizer`](corebehrt/data/tokenizer.py) + [`BatchTokenize`](corebehrt/data/batch.py)

Currently, still operates on sequences.  
Adds SEP and CLS tokens, add attention mask.  
Create dictionary based on pretrain_data

## 2. Pretrain Model
The pretraining process is handled by the [`main_pretrain.py`](corebehrt/main_pretrain.py) script. 

### [`DatasetPreparer`](corebehrt/data/prepare_data.py)
Loads in tokenized features (as patient sequences) and prepares them for pretraining.
1. Exclude short sequences (might be removed since happening during feature creation)
2. Optional: Select subset of patients
3. Truncation      
4. Normalize segments
Returns train and val [`MLMDataset`](corebehrt.data.dataset).

### [`Initializer`](corebehrt/common/initialize.py)
Initializes model, optimizer and scheduler.

### [`EHRTrainer`](corebehrt/trainer/trainer.py)
Responsible for training/monitoring/logging.

## 3. Create Outcomes (needed for fine-tuning)
The creation of outcomes is handled by the [`main_create_outcomes.py`](corebehrt/main_create_outcomes.py) script. This will extract events (censoring or target) from the raw data.  
[`OutcomeMaker`](corebehrt/downstream_tasks/outcomes.py)
Current implementation selects only first event in a patient timeline. Improved version will be available shortly which will allow for multiple events.

## 4. Fine-tune Model
The fine-tuning process is handled by the [`main_finetune_cv.py`](corebehrt/main_finetune_cv.py) script. This script uses the pre-trained model and fine-tunes it on the specific task. Finally, the model is tested on the test set.

### [`DatasetPreparer`](corebehrt/data/prepare_data.py)
1. Load in tokenized features (as patient sequences) and outcomes. *Will be expanded shortly with OutcomeHandler which will take more general exposures and outcomes and select based on index_date and start_of_follow_up*
2. Optional: Select gender
3. Optional: Select only exposed patients
4. Optional: Remove patients with outcome before index date *here we can also remove those with death before index date*
5. Optional: Filter code types+exclude short sequences *this can be removed*
6. Censoring
7. Selection based on age at index date *can be moved further up?*
8. Filter short sequences
9.  Truncation from the left
10. Normalize segments
11. Optional: Feature removal *can also be done in FeatureCreator*
12. Saving

### [`BinaryOutcomeDataset`](corebehrt/data/dataset/py)
Returns patient with target.

### [`ModelManager`](corebehrt/common/initialize.py)
Handling of loading in weights+config from pretrained model.  
Initializing training components.

### [`Initializer`](corebehrt/common/initialize.py)
Returning azure related stuff. Name should be changed/revamped to SDK2.

### [`EHRTrainer`](corebehrt/trainer/trainer.py)
Responsible for training/monitoring/logging.