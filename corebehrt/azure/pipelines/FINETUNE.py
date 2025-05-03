"""
Finetune pipeline implementation.
"""

from corebehrt.azure.pipelines.base import PipelineMeta, PipelineArg

FINETUNE = PipelineMeta(
    name="FINETUNE",
    help="Run the finetune pipeline.",
    inputs=[
        PipelineArg(name="data", help="Path to the raw input data.", required=True),
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

    @dsl.pipeline(name="finetune_pipeline", description="Finetune CoreBEHRT pipeline")
    def pipeline(
        data: Input,
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
        outcomes: Input = None,
    ) -> dict:
        if outcomes is None:
            create_outcomes = component(
                "create_outcomes",
            )(
                data=data,
                features=features,
            )
            outcomes = create_outcomes.outputs.outcomes

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

    return pipeline
