from corebehrt.azure import util

INPUTS = {"data": {"type": "uri_folder"}}
OUTPUTS = {
    "tokenized": {"type": "uri_folder"},
    "features": {"type": "uri_folder"},
}


def job(config, compute=None, register_output=dict(), log_system_metrics=False):
    return util.setup_job(
        "create_data",
        inputs=INPUTS,
        outputs=OUTPUTS,
        config=config,
        compute=compute,
        register_output=register_output,
        log_system_metrics=log_system_metrics,
    )


if __name__ == "__main__":
    from corebehrt.main import create_data

    util.run_main(create_data.main_data, INPUTS, OUTPUTS)
