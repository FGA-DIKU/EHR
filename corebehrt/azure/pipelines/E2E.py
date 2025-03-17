from corebehrt.azure.util import check_azure
from corebehrt.azure.util.pipeline import create_component


def create(
    configs: dict,
    computes: dict,
    register_output: dict,
    log_system_metrics: bool,
    test_cfg_file: str = None,
) -> "pipeline":  # noqa: F821

    check_azure()
    from azure.ai.ml import dsl, Input

    def component(job_name: str, name: str = None):
        return create_component(
            job_name,
            configs,
            computes,
            register_output,
            log_system_metrics,
            test_cfg_file,
            name=name,
        )

    @dsl.pipeline(description="Full E2E CoreBEHRT pipeline")
    def pipeline(data: Input) -> dict:
        create_data = component(
            "create_data",
        )(data=data)

        create_outcomes = component(
            "create_outcomes",
        )(
            data=data,
            features=create_data.outputs.features,
        )

        select_cohort = component(
            "select_cohort",
        )(
            features=create_data.outputs.features,
            outcomes=create_outcomes.outputs.outcomes,
        )

        prepare_pretrain = component(
            "prepare_training_data",
            name="prepare_pretrain",
        )(
            features=create_data.outputs.features,
            tokenized=create_data.outputs.tokenized,
        )

        pretrain = component("pretrain")(
            prepared_data=prepare_pretrain.outputs.prepared_data,
        )

        prepare_finetune = component(
            "prepare_training_data",
            name="prepare_finetune",
        )(
            features=create_data.outputs.features,
            tokenized=create_data.outputs.tokenized,
            cohort=select_cohort.outputs.cohort,
            outcomes=create_outcomes.outputs.outcomes,
        )

        finetune = component(
            "finetune_cv",
        )(
            prepared_data=prepare_finetune.outputs.prepared_data,
            pretrain_model=pretrain.outputs.model,
        )

        return {
            "pretrain_model": pretrain.outputs.model,
            "model": finetune.outputs.model,
        }

    return pipeline
