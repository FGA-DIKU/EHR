from os.path import exists, join
import pandas as pd

from corebehrt.main.finetune_cv import main_finetune
from corebehrt.main.evaluate_finetune import main_evaluate
from tests.test_main.base import TestMainScript


class TestFinetune(TestMainScript):
    def setUp(self):
        super().setUp()

        # Paths
        self.processed_dir = "./tests/data/finetune/processed_data"
        self.pretrain_model = "./tests/data/pretrain"
        self.model_dir = join(self.tmp_dir, "finetune")
        self.test_data = "./tests/data/testing/processed_data"
        self.predictions = join(self.tmp_dir, "testing/predictions")

        self.set_config(
            {
                "paths": {
                    "prepared_data": self.processed_dir,
                    "pretrain_model": self.pretrain_model,
                    "model": self.model_dir,
                },
                "model": {
                    "cls": "default",
                },
                "trainer_args": {
                    "batch_size": 128,
                    "val_batch_size": 128,
                    "effective_batch_size": 128,
                    "epochs": 30,
                    "info": True,
                    "gradient_clip": {
                        "clip_value": 1.0,
                    },
                    "sampler_function": {
                        "_target_": "corebehrt.modules.trainer.utils.Sampling.inverse_sqrt",
                    },
                    "loss_weight_function": {
                        "_target_": "corebehrt.modules.trainer.utils.PositiveWeight.sqrt",
                    },
                    "checkpoint_frequency": 1,
                    "early_stopping": 5,
                    "stopping_criterion": "roc_auc",
                    "n_layers_to_freeze": 1,
                    "unfreeze_on_plateau": True,
                    "unfreeze_at_epoch": 2,
                    "plateau_threshold": 0.01,
                    "reset_patience_after_unfreeze": True,
                },
                "optimizer": {
                    "lr": 5e-4,
                    "eps": 1e-6,
                },
                "scheduler": {
                    "_target_": "transformers.get_linear_schedule_with_warmup",
                    "num_warmup_steps": 10,
                    "num_training_steps": 100,
                },
            }
        )

    def setup_eval_config(self):
        eval_config = {
            "paths": {
                "model": self.model_dir,
                "folds_dir": self.processed_dir,
                "test_data_dir": self.processed_dir,
                "run_name": "finetune",
                "predictions": self.predictions,
            },
            "test_batch_size": 128,
            "return_embeddings": True,
            "metrics": {
                "roc_auc": {
                    "_target_": "sklearn.metrics.roc_auc_score",
                },
                "pr_auc": {
                    "_target_": "sklearn.metrics.average_precision_score",
                },
            },
        }
        self.set_config(eval_config)

    def test_finetune(self):
        ### Call finetune script
        main_finetune(self.config_path)

        ### Validate generated files

        # 1: Check that the finetuned model directory exists
        self.assertTrue(exists(self.model_dir))

        # 2: Check that the model files are created
        model_files = [
            "checkpoints",
            "finetune_config.yaml",
            "pretrain_config.yaml",
            "fold_1",
        ]
        for file_name in model_files:
            path = join(self.model_dir, file_name)
            self.assertTrue(exists(path), f"Missing file: {file_name}")

        self.setup_eval_config()
        print(self.config_path)
        main_evaluate(self.config_path)
        eval_files = [
            "predictions.csv",
            "metrics.csv",
            "embeddings/fold_1/attention_masks.pt",
            "embeddings/fold_1/BERT_embeddings.pt",
            "embeddings/fold_1/cls_embeddings.pt",
            "embeddings/fold_1/pids.pt",
            "evaluate_finetune.yaml",
            "finetune_config.yaml",
        ]
        for file_name in eval_files:
            path = join(self.predictions, file_name)
            self.assertTrue(exists(path), f"Missing file: {file_name}")

        # Check that the metrics are correct
        metrics = pd.read_csv(join(self.predictions, "metrics.csv"))
        print(metrics)
        rocs = metrics["roc_auc"].tolist()
        prs = metrics["pr_auc"].tolist()
        self.assertTrue(all(roc >= 0.7 for roc in rocs))
        self.assertTrue(all(pr >= 0.4 for pr in prs))
