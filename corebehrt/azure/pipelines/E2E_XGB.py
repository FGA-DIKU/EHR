"""
E2E pipeline implementation.
"""

from corebehrt.azure.pipelines.base import PipelineMeta, PipelineArg

E2E_XGB = PipelineMeta(
    name="E2E_XGB",
    help="Run the end-to-end pipeline with XGBoost.",
    inputs=[
        PipelineArg(name="data", help="Path to the raw input data.", required=True),
    ],
)


def create(component: callable):
    """
    Define the E2E pipeline.

    Param component(job_type, name=None) is a constructor for components
    which takes arguments job_type (type of job) and optional argument
    name (name of component if different from type of job).
    """
    from azure.ai.ml import dsl, Input

    @dsl.pipeline(name="E2E_XGB_pipeline", description="Full E2E XGBoost pipeline")
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

        prepare_finetune = component(
            "prepare_training_data",
            name="prepare_finetune",
        )(
            features=create_data.outputs.features,
            tokenized=create_data.outputs.tokenized,
            cohort=select_cohort.outputs.cohort,
            outcomes=create_outcomes.outputs.outcomes,
        )

        xgboost = component(
            "xgboost_cv",
        )(
            prepared_data=prepare_finetune.outputs.prepared_data,
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

        evaluate_xgboost = component(
            "evaluate_xgboost",
        )(
            model=xgboost.outputs,
            folds_dir=prepare_finetune.outputs.prepared_data,
            test_data_dir=prepare_held_out.outputs.prepared_data,
        )

        return {
            "model": xgboost.outputs.model,
            "predictions": evaluate_xgboost.outputs.predictions,
        }

    return pipeline
