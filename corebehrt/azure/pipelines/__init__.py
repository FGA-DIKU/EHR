from corebehrt.azure.pipelines.E2E import E2E
from corebehrt.azure.pipelines.E2E_full import E2E_full
from corebehrt.azure.pipelines.FINETUNE import FINETUNE


PIPELINE_REGISTRY = [E2E, E2E_full, FINETUNE]
