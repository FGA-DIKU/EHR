# Running CoreBEHRT in Azure with SDK v2

## Configuration files in Azure
The normal configuration files can be used when running in Azure with only minor changes to the `paths` sub-configuration. Specifically, paths must be either paths on data stores (`research-data` or `sp-data`) or data asset identifiers:

A path on the data store must be specified as `<datastore>:<path>`, while an asset must be specified as `<asset_name>:<asset_version>` or `<asset_name>@<asset_label>`.

Example from `create_data.yaml`, using the CoreBEHRT example data directory registered Azure asset (`CoreBEHRT_example_data`, latest version) as input, and folders on `researcher-data` for output:
```
paths:
    data: "CoreBEHRT_example_data@latest"
    features: "researher-data:unit_tests/corebehrt/output/features"
    tokenized: "researher-data:unit_tests/corebehrt/output/tokenized"
```

## Running from the command line
The `azure` submodule can be run directly from the command line:
```
python -m corebehrt.azure [-h] {build_env, job}
```

### Building the CoreBEHRT environment
The `CoreBEHRT` Azure environment needed for running CoreBEHRT jobs can be built using the sub-command
```
python -m corebehrt.azure build_env
```
The environment must be build, before jobs can be run.

### Running jobs
CoreBEHRT jobs are run using the `job` sub-command:
```
python -m corebehrt.azure job {create_data,create_outcomes,pretrain,finetune_cv} <compute> [-e <experiment>] [-o <output_id>=<output_name>] [-o ...] [-c <path_to_config>]
```
The command starts the specified job using the specified `experiment` and `compute`. Passing pairs `<ouput_id>=<output_name>` using `-o` allows for registering outputs as data assets. The default configuration path (`corebehrt/configs/<job_name>.yaml`) can be overridden with the `-c`/`--config` option.

Example of running `create_data` and registering outputs:
```
corebehrt.azure -e "CBTest" -o features=CBFeatures -o tokenized=CBTokenized CPU-20-LP job create_data -c create_data_on_azure.yaml
```

## Running from a python script.

### Building the CoreBEHRT environment
The CoreBEHRT environment can be build from a script/notebook using:
```
from corebehrt.azure import environment

environment.build()
```

### Running jobs
Each CoreBEHRT main can be run from a script/notebook as well:
```
from corebehrt.azure import util
from corebehrt.azure.components import <main_module>

cfg = load_config(<path_to_cfg>)
job = <main_module>.job(cfg, compute=<compute>, register_output=<output_mapping>)
util.run_job(job, <experiment>)
```
where `<main_module>` is `create_data`, `create_outcomes`, `pretrain` or `finetune_cv`.

