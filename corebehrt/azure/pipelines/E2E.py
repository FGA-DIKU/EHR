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
    def pipeline(data: Input) -> dict:
        create_data = create_component(
            "create_data", configs, computes, register_output, log_system_metrics
        )(data=data)

        create_outcomes = create_component(
            "create_outcomes", configs, computes, register_output, log_system_metrics
        )(
            data=data,
            features=create_data.outputs.features,
        )

        select_cohort = create_component(
            "select_cohort",
            configs,
            computes,
            register_output,
            log_system_metrics,
        )(
            features=create_data.outputs.features,
            outcomes=create_outcomes.outputs.outcomes,
        )

        prepare_pretrain = create_component(
            "prepare_training_data",
            configs,
            computes,
            register_output,
            log_system_metrics,
        )(
            features=create_data.outputs.features,
            tokenized=create_data.outputs.tokenized,
            cohort=select_cohort.outputs.cohort,
        )

        pretrain = create_component(
            "pretrain", configs, computes, register_output, log_system_metrics
        )(
            prepared_data=prepare_pretrain.outputs.prepared_data,
        )

        return {}

    return pipeline
