# Dask Performance Tests

This directory contains performance benchmarking tests for different Dask DataFrame operations, specifically focused on various filtering strategies and their performance implications.

## Test Files

### 1. Filtering Direct vs Partitions (`filtering_direct_vs_partitions.py`)

Compares two approaches for filtering patient data:

- Direct filtering using `.isin()`
- Partition-based filtering using `map_partitions` with a precomputed set

This test helps understand the performance implications of different filtering strategies when working with large patient datasets distributed across multiple partitions.

### 2. Filtering Index vs isin (`filtering_index_vs_isin.py`)

Compares two methods for filtering data by patient IDs:

- Using `set_index` followed by `loc` indexing
- Using the `.isin()` method directly

Includes configurable parameters for:

- Number of test runs
- Number of patients
- Events per patient

### 3. Data Generation (`generate_features.py`)

Contains utilities for generating synthetic healthcare data with:

- Patient IDs (PID)
- Age values
- Event types
- Timestamps

The generated data follows realistic patterns with:

- Variable number of events per patient (exponential distribution)
- Age distribution between 20-90 years
- Multiple event types (A, B, C, D, E)

## Usage

Each test can be run independently. For example:

```bash
python filtering_direct_vs_partitions.py
```

```bash
python filtering_index_vs_isin.py
```
