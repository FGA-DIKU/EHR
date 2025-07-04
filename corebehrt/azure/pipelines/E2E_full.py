"""
E2E_full pipeline implementation.
"""

from corebehrt.azure.pipelines.base import PipelineMeta, PipelineArg

E2E_full = PipelineMeta(
    name="E2E_full",
    help="Run the end-to-end pipeline with held out data.",
    inputs=[
        PipelineArg(name="data", help="Path to the raw input data.", required=True),
    ],
)


def create(component: callable):
    """
    Define the E2E_full pipeline.

    Param component(job_type, name=None) is a constructor for components
    which takes arguments job_type (type of job) and optional argument
    name (name of component if different from type of job).
    """
    from azure.ai.ml import dsl, Input

    @dsl.pipeline(name="E2E_pipeline_full", description="Full E2E CoreBEHRT pipeline")
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

        select_cohort_held_out = component(
            "select_cohort",
            name="select_held_out_cohort",
        )(
            features=create_data.outputs.features,
            outcomes=create_outcomes.outputs.outcomes,
        )

        prepare_held_out = component(
            "prepare_training_data",
            name="prepare_held_out",
        )(
            features=create_data.outputs.features,
            tokenized=create_data.outputs.tokenized,
            cohort=select_cohort_held_out.outputs.cohort,
            outcomes=create_outcomes.outputs.outcomes,
        )

        evaluate_finetune = component(
            "evaluate_finetune",
        )(
            model=finetune.outputs.model,
            folds_dir = prepare_finetune.outputs.prepared_data,
            test_data_dir = prepare_held_out.outputs.prepared_data,
        )

        return {
            "pretrain_model": pretrain.outputs.model,
            "model": finetune.outputs.model,
            "predictions": evaluate_finetune.outputs.predictions,
        }

    return pipeline
