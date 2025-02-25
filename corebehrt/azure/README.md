# Running CoreBEHRT in Azure with SDK v2

## Setting up the instance environment

The commands listed below must be run from an Azure compute instance. The `corebehrt.azure` sub-module requires only three packages to run:

```
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

