# Running CoreBEHRT in Azure with SDK v2

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
python -m corebehrt.azure job {create_data,create_outcomes,pretrain,select_cohort,finetune_cv} <compute> [-e <experiment>] [-o <output_id>=<output_name>] [-o ...] [-c <path_to_config>]
```

Here are more examples of running different job types:

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

The command starts the specified job using the specified `experiment` and `compute`. Passing pairs `<ouput_id>=<output_name>` using `-o` allows for registering outputs as data assets. The default configuration path (`corebehrt/configs/<job_name>.yaml`) can be overridden with the `-c`/`--config` option.

Example of running `create_data` and registering outputs:

```bash
corebehrt.azure -e "CBTest" -o features=CBFeatures -o tokenized=CBTokenized CPU-20-LP job create_data -c create_data_on_azure.yaml
```

## Running from a python script

### Building the CoreBEHRT environment (script)

The CoreBEHRT environment can be build from a script/notebook using:

```python
from corebehrt.azure import environment

environment.build()
```

### Azure configuration

When running on Azure, you should create/modify an `.amlignore` file in the root folder to exclude unnecessary directories from being uploaded to the compute instance. Create the file with the following contents:

```text
tests/
example_data/
notebooks/
```

This will speed up job submission, and reduce unnecessary data transfer.

### Running jobs (script)

Each CoreBEHRT main can be run from a script/notebook as well:

```python

from corebehrt.azure import util
from corebehrt.azure.components import <main_module>

cfg = load_config(<path_to_cfg>)
job = <main_module>.job(cfg, compute=<compute>, register_output=<output_mapping>)
util.run_job(job, <experiment>)
```

where `<main_module>` is `create_data`, `create_outcomes`, `pretrain` or `finetune_cv`.

