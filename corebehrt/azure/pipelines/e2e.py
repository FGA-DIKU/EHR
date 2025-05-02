"""
E2E pipeline implementation.
"""

from corebehrt.azure.pipelines.base import PipelineMeta

E2E = PipelineMeta(
    name="E2E",
    help="Run the end-to-end pipeline.",
    required_inputs={
        "data": {"help": "Path to the raw input data."},
    },
    helper_inputs={
        # "foo": {"help": "Optional helper input."},
    },
)


def create(component: callable):
    """
    Define the E2E pipeline.

    Param component(job_type, name=None) is a constructor for components
    which takes arguments job_type (type of job) and optional argument
    name (name of component if different from type of job).
    """
    from azure.ai.ml import dsl, Input

    @dsl.pipeline(name="E2E", description="Full E2E CoreBEHRT pipeline")
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
