# Running CoreBEHRT in Azure with SDK v2

## Setting up the instance environment

The commands listed below must be run from an Azure compute instance. The `corebehrt.azure` sub-module requires only three packages to run:

```text
pyyaml
azure-identity
azure-ai-ml
```

These are usually installed in the default environment, once a new compute instance is created.

The environment on the cluster can be set up using [the `build_env` command](#building-the-corebehrt-environment).

### Ignoring files when uploading jobs to cluster

When running on Azure, you should create/modify an `.amlignore` file in the root folder to exclude unnecessary directories from being uploaded to the compute instance. Create the file with the following contents:

```text
tests/
example_data/
notebooks/
```

This will speed up job submission, and reduce unnecessary data transfer.

## Configuration files in Azure

The normal configuration files can be used when running in Azure with only minor changes to the `paths` sub-configuration. Specifically, paths must be either paths on data stores (`research-data` or `sp-data`) or data asset identifiers:

A path on the data store must be specified as `<datastore>:<path>`, while an asset must be specified as `<asset_name>:<asset_version>` or `<asset_name>@<asset_label>`.

Example from `create_data.yaml`, using the CoreBEHRT example data directory registered Azure asset (`CoreBEHRT_example_data`, latest version) as input, and folders on `researcher-data` for output:

```yaml
paths:
    data: "CoreBEHRT_example_data@latest"
    features: "researher-data:unit_tests/corebehrt/output/features"
    tokenized: "researher-data:unit_tests/corebehrt/output/tokenized"
```

## Running from the command line

The `azure` submodule can be run directly from the command line:

```bash
python -m corebehrt.azure [-h] {build_env, job}
```

### Building the CoreBEHRT environment

The `CoreBEHRT` Azure environment needed for running CoreBEHRT jobs can be built using the sub-command:

```bash
python -m corebehrt.azure build_env
```

The environment must be build, before jobs can be run.

### Running jobs

CoreBEHRT jobs are run using the `job` sub-command:

```bash
python -m corebehrt.azure job {create_data,create_outcomes,pretrain,select_cohort,finetune_cv} <compute> [-e <experiment>] [-o <output_id>=<output_name>] [-o ...] [-c <path_to_config>] [--log_system_metrics]
```

The command starts the specified job using the specified `<experiment>` and `<compute>`. Passing pairs `<ouput_id>=<output_name>` using `-o` allows for registering outputs as data assets. The default configuration path (`corebehrt/configs/<job_name>.yaml`) can be overridden with the `-c`/`--config` option. If set, the `--log_system_metrics` (alt. `-lsm`) enables logging of CPU, GPU and memory utilization.

Examples of running different job types:

```bash
# Create data using CPU compute
python -m corebehrt.azure job create_data CPU-20-LP -e test_pipeline -c azure_configs/create_data.yaml

# Run pretraining on GPU
python -m corebehrt.azure job pretrain GPU-A100-Single -e test_pipeline -c azure_configs/pretrain.yaml

# Create outcomes using CPU compute
python -m corebehrt.azure job create_outcomes CPU-20-LP -e test_pipeline -c azure_configs/outcome_mace.yaml

# Select cohort using CPU compute
python -m corebehrt.azure job select_cohort CPU-20-LP -e test_pipeline -c azure_configs/select_cohort.yaml

# Run fine-tuning with cross-validation on GPU
python -m corebehrt.azure job finetune_cv GPU-A100-Single -e test_pipeline -c azure_configs/finetune.yaml
```

Example of running `create_data` and registering outputs:

```bash
python -m corebehrt.azure job create_data CPU-20-LP -e "CBTest" -o features=CBFeatures -o tokenized=CBTokenized -c create_data_on_azure.yaml
```

### Running pipelines

CoreBEHRT pipelines are currently added in `corebehrt.azure.pipelines`. The following pipelines are available:

- **E2E Pipeline**: Complete end-to-end pipeline including pretraining and finetuning

```bash
python -m corebehrt.azure pipeline E2E -i data=<path> [<default-compute>] [<config-dir>] [-cp <component-name>=<compute>, +] [-c <component-name>=<config-path>] -e <experiment>
```

- **Finetune Pipeline**: Pipeline for finetuning a pretrained model

```bash
python -m corebehrt.azure pipeline finetune -i data=<path> -i features=<path> -i tokenized=<path> -i pretrained_model=<path> ... (see E2E pipeline)
```

#### Pipeline Components

**E2E Pipeline Components:**

- `create_data`
- `create_outcomes`
- `select_cohort`
- `prepare_pretrain`
- `prepare_finetune`
- `pretrain`
- `finetune_cv`

**Finetune Pipeline Components:**

- `create_outcomes`
- `select_cohort`
- `prepare_finetune`
- `finetune_cv`

`<config-dir>` must contain a config file for each component in the chosen pipeline. Options `-cp` and `-c` are used to overwrite computes and config paths respectively, for individual components.

#### Examples

Running E2E on example MEDS data:

```bash
python -m corebehrt.azure pipeline E2E -i data=CoreBEHRT_example_data@latest CPU-20-LP corebehrt/azure/configs/small -cp pretrain=GPU-A100-Single -cp finetune_cv=GPU-A100-Single -e full_e2e_test
```

Running finetune with a pretrained model:

```bash
python -m corebehrt.azure pipeline finetune -i data=CoreBEHRT_example_data@latest -i pretrained_model="azureml://jobs/<pretrain-job-id>/outputs/model" -i features=CBFeatures -i tokenized=CBTokenized CPU-20-LP corebehrt/azure/configs/small -cp finetune_cv=GPU-A100-Single -e finetune_test
```

This uses `CPU-20-LP` as the default compute, but uses `GPU-A100-Single` for compute-intensive components.

**Note on configs for pipelines:** Input/output configs for pipelines, contrary to configs for singular jobs, may leave out paths for inputs, as these are always tied to an output from another component. Output paths may be left out (in which case a location in the default blobstore is created).

One exception is for input paths, which does not exactly correspond to the output of another component, e.g. `outcome` for `prepare_finetune` (which is of job type `prepare_training_data`). `outcome` references a file in the `outcomes` directory produced by `create_outcomes`. A proper config file must set `outcome` to the path to this file, **relative to the `outcomes` directory**.

See the pipeline example configs in `corebehrt/azure/configs`.

**Note on adding more pipelines**: This is mostly added for creating full pipeline tests. The command line and utility functions currently only support pipelines with a single input (called `data`). Adding pipelines with additional inputs (e.g. `predefined_splits`) requires more work and should probably be done as a more flexible and configuration file based setup (similar to the job setup).

### Running tests

A full test of all components chained in the E2E pipeline can be run using:

```bash
python -m corebehrt.azure test <test-name>
```

This will run the given test `<test-name>` in the experiment `corebehrt_pipeline_tests`. Each tests must have a set of valid component configs + a test config in a properly named directory in `corebehrt/azure/configs`. The test config specifies:

- `data`: data set to use.
- `computes`: computes to use for each step, with `computes.default` specifying the default.
- `<component-name>`: A section for each component specifying tests related to that component:
  - `max_run_time`: Max run time given in seconds.
  - `metrics`: A list of elements with attributes `type` (name of metric), `child` (name of sub-run, e.g. "Fold 1" or "val scores"), `min` (optional minimum value), `max` (optional maximum value).
  - `on_fail`: Optional, if set to `raise`, a test failure will raise an exception and halt the pipeline. If not set, test results will only be logged.

Note, that long running components will not be halted after `max_run_time` seconds has passed. The check is only made after the component has finished. Thus, for bugs increasing runtime to several days, jobs should simply be cancelled.

See the `test.yaml` files in subdirectories of `corebehrt/azure/configs` for examples.

Available tests are:

- **small**: Runs E2E on `example_data/example_MEDS_data` (`CoreBEHRT_example_data@latest`).
- **full**: Runs E2E on `MEDS_all_20240910:1`.

## Running from a python script

### Building the CoreBEHRT environment (script)

The CoreBEHRT environment can be build from a script/notebook using:

```python
from corebehrt.azure import environment

environment.build()
```

### Running jobs (script)

Each CoreBEHRT main can be run from a script/notebook as well:

```python

from corebehrt.azure import create_job, run_job
from corebehrt.modules.setup.config import load_config

cfg = load_config(<path_to_cfg>).to_dict()
job = create_job(<job_name>, cfg, compute=<compute>, register_output=<output_mapping>, log_system_metrics=<log_system_metrics>)
util.run_job(job, <experiment>)
```

where `<job_name>` is one of the CoreBEHRT main scripts, `<compute>` is the name of the compute cluster to use, `<register_output>` (optional) is a dict mapping output names to asset names, and `<log_system_metrics>` (optional, default is `False`) is a boolean.
