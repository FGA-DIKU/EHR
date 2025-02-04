# Detailed Module Structure

## Cohort Handling

### `index_dates.py`

Handles creation and management of index dates for patient cohorts.

- **`IndexDateHandler`**: Creates and manages timestamp series for patients.
- Supports absolute and relative index date modes.

### `outcomes.py`

Manages outcome data for patients.

- **`OutcomeMaker`**: Creates outcomes from concepts and patient info.
- Handles different types of outcomes (patient info, concepts).

### `patient_filter.py`

Filters patients based on various criteria.

- Filters by categories, age ranges.
- Handles exclusion based on death and outcomes.

## Features

### `excluder.py`

Excludes incorrect events and patients.

- **`Excluder`**: Handles age-based event exclusion.

### `features.py`

Core feature creation functionality.

- **`FeatureCreator`**: Creates features from patient information.
- Handles background, death, age, and position features.

### `normalizer.py`

Normalizes values in data frames.

- **`ValuesNormalizer`**: Performs min-max normalization on results.

### `tokenizer.py`

Handles tokenization of medical concepts.

- **`EHRTokenizer`**: Manages vocabulary and tokenization.
- Handles special tokens (`CLS`, `SEP`, `MASK`, etc.).

### `values.py`

Creates and manages feature values.

- **`ValueCreator`**: Handles binned and quantile values.

## Model

### `embeddings.py`

Neural network embedding layers.

- **`EhrEmbeddings`**: Custom embeddings for EHR data.
- **`Time2Vec`**: Time-based embedding implementation.

### `heads.py`

Model head implementations.

- **`MLMHead`**: Masked Language Model head.
- **`FineTuneHead`**: Fine-tuning head with BiGRU.
- **`BiGRU`**: Bidirectional GRU implementation.

### `model.py`

Core model implementations.

- **`BertEHREncoder`**: Base BERT encoder for EHR data.
- **`BertForFineTuning`**: Fine-tuning model implementation.

## Preparation

### `dataset.py`

Dataset classes for training.

- **`PatientData`**: Data structure for patient information.
- **`PatientDataset`**: Dataset management and processing.
- **`MLMDataset`**: Dataset for masked language modeling.
- **`BinaryOutcomeDataset`**: Dataset for binary outcomes.

### `mask.py`

Masking functionality for MLM.

- **`ConceptMasker`**: Handles concept masking for training.

### `prepare_data.py`

Data preparation pipeline.

- **`DatasetPreparer`**: Prepares datasets for pre-training and fine-tuning.

## Setup

### `config.py`

Configuration management.

- **`Config`**: Configuration class with dot notation.
- Functions for loading and instantiating from config.

### `directory.py`

Directory structure management.

- **`DirectoryPreparer`**: Manages directory setup and validation.
- Handles config file management and logging setup.

## Monitoring

### `logger.py`

Logging utilities.

- **`TqdmToLogger`**: Progress bar logging.
- Progress tracking utilities.

### `metrics.py`

Model evaluation metrics.

- Various metric implementations (accuracy, precision, recall, etc.).
- **`BaseMetric`**: Base class for metric implementations.

### `metric_aggregation.py`

Metric aggregation utilities.

- Functions for computing and saving metric averages.
- Handles prediction saving and metric computation.
