from corebehrt.azure.util import check_azure
from corebehrt.azure.util.pipeline import create_component

INPUTS = {"data": {"type": "uri_folder", "config": "create_data"}}
OUTPUTS = {"model": {"type": "uri_folder", "config": "pretrain"}}


def create(
    configs: dict, computes: dict, register_output: dict, log_system_metrics: bool
) -> "pipeline":  # noqa: F821

    check_azure()
    from azure.ai.ml import dsl, Input

    @dsl.pipeline(description="Full E2E CoreBEHRT pipeline")
    def pipeline(data: Input):
        prepare_data = create_component(
            "create_data", configs, computes, register_output, log_system_metrics
        )(data=data)

        pretrain = create_component(
            "pretrain", configs, computes, register_output, log_system_metrics
        )(
            features=prepare_data.outputs.features,
            tokenized=prepare_data.outputs.tokenized,
        )

        return {"model": pretrain.outputs.model}

    return pipeline
