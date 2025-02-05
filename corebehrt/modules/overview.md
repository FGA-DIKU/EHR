# CoreBEHRT Modules Overview

## Overview by Submodule

### Cohort Handling

Manages patient cohorts, outcomes, and filtering. Responsible for:

- Creating and managing patient timelines and index dates
- Defining and tracking patient outcomes
- Filtering patients based on clinical criteria
- Managing temporal aspects of patient data

### Features

Handles all feature-related operations including:

- Event and patient exclusion logic
- Feature creation and normalization
- Tokenization of medical concepts
- Value binning and quantile calculations
- Data normalization and standardization

### Model

Contains core model architecture and components:

- Custom embedding layers for EHR data
- Model heads for different tasks (MLM, fine-tuning)
- BERT-based encoder implementation
- Time-based embeddings and position encoding
- Task-specific model variants

### Preparation

Manages data preparation pipeline:

- Dataset creation and management
- Masking strategies for pre-training
- Data preprocessing and transformation
- Batch preparation and sampling
- Dataset splitting and validation

### Setup

Handles configuration and initialization:

- Configuration management and validation
- Directory structure setup
- Environment preparation
- Logging configuration
- File path management

### Monitoring

Manages tracking and evaluation:

- Logging and progress tracking
- Metric calculation and aggregation
- Model evaluation
- Performance monitoring
- Result storage and analysis
