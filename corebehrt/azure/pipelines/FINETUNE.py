"""
Finetune pipeline implementation with proper handling of optional inputs.
"""

from corebehrt.azure.pipelines.base import PipelineMeta, PipelineArg

FINETUNE = PipelineMeta(
    name="FINETUNE",
    help="Run the finetune pipeline.",
    inputs=[
        PipelineArg(
            name="data",
            help="Path to the raw input data. Only required if outcomes is not provided.",
            required=False,
        ),
        PipelineArg(name="features", help="Path to the features data.", required=True),
        PipelineArg(
            name="tokenized", help="Path to the tokenized data.", required=True
        ),
        PipelineArg(
            name="pretrain_model", help="Path to the pretrained model.", required=True
        ),
        PipelineArg(name="outcomes", help="Path to the outcomes data.", required=False),
    ],
)


def create(component: callable):
    """
    Define the Finetune pipeline.

    Param component(job_type, name=None) is a constructor for components
    which takes arguments job_type (type of job) and optional argument
    name (name of component if different from type of job).
    """
    from azure.ai.ml import dsl, Input
    from typing import Dict, Any

    # Define a factory function that returns the appropriate pipeline
    def pipeline_factory(**kwargs: Dict[str, Any]):
        # Check if outcomes is provided
        has_outcomes = "outcomes" in kwargs and kwargs["outcomes"] is not None

        if has_outcomes:
            # With outcomes, we don't need data
            if "data" in kwargs:
                del kwargs["data"]
            return _pipeline_with_outcomes(**kwargs)
        else:
            # Without outcomes, we need data
            if "data" not in kwargs:
                raise ValueError(
                    "'data' input is required when 'outcomes' is not provided"
                )

            # Remove outcomes from kwargs if it exists but is None
            if "outcomes" in kwargs:
                del kwargs["outcomes"]
            return _pipeline_without_outcomes(**kwargs)

    @dsl.pipeline(
        name="finetune_with_outcomes",
        description="Finetune CoreBEHRT pipeline with provided outcomes",
    )
    def _pipeline_with_outcomes(
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
        outcomes: Input,
    ) -> dict:
        # Using provided outcomes
        select_cohort = component(
            "select_cohort",
        )(
            features=features,
            outcomes=outcomes,
        )

        prepare_finetune = component(
            "prepare_training_data",
            name="prepare_finetune",
        )(
            features=features,
            tokenized=tokenized,
            cohort=select_cohort.outputs.cohort,
            outcomes=outcomes,
        )

        finetune = component(
            "finetune_cv",
        )(
            prepared_data=prepare_finetune.outputs.prepared_data,
            pretrain_model=pretrain_model,
        )

        return {
            "model": finetune.outputs.model,
        }

    @dsl.pipeline(
        name="finetune_without_outcomes",
        description="Finetune CoreBEHRT pipeline with auto-created outcomes",
    )
    def _pipeline_without_outcomes(
        data: Input,
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
    ) -> dict:
        # Create outcomes internally
        create_outcomes = component(
            "create_outcomes",
        )(
            data=data,
            features=features,
        )

        select_cohort = component(
            "select_cohort",
        )(
            features=features,
            outcomes=create_outcomes.outputs.outcomes,
        )

        prepare_finetune = component(
            "prepare_training_data",
            name="prepare_finetune",
        )(
            features=features,
            tokenized=tokenized,
            cohort=select_cohort.outputs.cohort,
            outcomes=create_outcomes.outputs.outcomes,
        )

        finetune = component(
            "finetune_cv",
        )(
            prepared_data=prepare_finetune.outputs.prepared_data,
            pretrain_model=pretrain_model,
        )

        return {
            "model": finetune.outputs.model,
        }

    # Return the factory function instead of a direct pipeline
    return pipeline_factory
