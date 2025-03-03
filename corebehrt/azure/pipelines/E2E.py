from ..util import check_azure, create_component


def create(
    configs: dict, computes: dict, register_output: dict, log_system_metrics: bool
) -> "pipeline":  # noqa: F821

    check_azure()
    from azure.ai.ml import dsl, Input, Output

    @dsl.pipeline(description="Full E2E CoreBEHRT pipeline")
    def pipeline(raw_ehr_data):
        prepare_data = create_component(
            "create_data", configs, computes, register_output, log_system_metrics
        )(data=raw_ehr_data)

        pretrain = create_component(
            "pretrain", configs, computes, register_output, log_system_metrics
        )(
            features=prepare_data.outputs.features,
            tokenized=prepare_data.outputs.tokenized,
        )

        return {"model": pretrain.outputs.model}

    return pipeline
