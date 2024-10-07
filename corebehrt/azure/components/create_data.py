from corebehrt.azure import util
from corebehrt.main import create_data
import argparse

INPUTS = {
    "paths": {"save_features_dir_name": {"type": str, "default": "features"}},
    "loader": {
        "data_dir": {
            "type": "uri_folder",
        },
        "concepts": {
            "type": str,
            "action": "append",
            "default": [],
        },
        "batchsize": {
            "type": int,
            "default": 64,
        },
        "chunksize": {"type": int, "default": 300},
    },
    "features": {
        "background_vars": {
            "type": str,
            "action": "append",
            "default": [],
        },
    },
    "tokenizer": {
        "sep_tokens": {
            "type": bool,
            "default": True,
        },
        "cls_token": {
            "type": bool,
            "default": True,
        },
    },
    "excluder": {
        "min_len": {
            "type": int,
            "default": 2,
        },
        "min_age": {
            "type": int,
            "default": -1,
        },
        "max_age": {
            "type": int,
            "default": 120,
        },
    },
    "split_ratios": {
        "pretrain": {
            "type": float,
            "default": 0.72,
        },
        "finetune": {
            "type": float,
            "default": 0.18,
        },
        "test": {
            "type": float,
            "default": 0.1,
        },
    },
    "tokenized_dir_name": {"type": str, "default": "tokenized"},
}
OUTPUTS = {"output_dir": {"type": "uri_folder"}}


def component():
    return util.setup_component(ARGS)


def job(config, **kwargs):
    kwargs["compute"] = kwargs.get("compute", "CPU-20-LP")
    return util.setup_job(
        "create_data",
        job="create_data",
        inputs=INPUTS,
        outputs=OUTPUTS,
        config=config,
        **kwargs,
    )


if __name__ == "__main__":
    # Parse inputs and run command
    cfg_path = util.parse_config("create_data", INPUTS | OUTPUTS, save_to_dir="./logs")
    create_data.main_data(cfg_path)
