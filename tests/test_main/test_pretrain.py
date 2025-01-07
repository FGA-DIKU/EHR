from os.path import getmtime, join, exists

from corebehrt.main.pretrain import main_train
from corebehrt.common.setup import DATA_CFG, PRETRAIN_CFG, PROCESSED_DATA_DIR

from .base import TestMainScript


class TestCreateOutcomes(TestMainScript):
    def setUp(self):
        super().setUp()

        self.pretrain_dir = join(self.tmp_dir, "pretrain")

        self.set_config(
            {
                "paths": {
                    "data": "./tests/data/raw",
                    "features": "./tests/data/features",
                    "tokenized": "./tests/data/tokenized",
                    "model": self.pretrain_dir,
                },
                "save_processed_data": True,
                "data": {
                    "dataset": {
                        "select_ratio": 1.0,
                        "masking_ratio": 0.8,
                        "replace_ratio": 0.1,
                        "ignore_special_tokens": True,
                    },
                    "truncation_len": 20,
                    "num_train_patients": 100,
                    "num_val_patients": 20,
                    "val_ratio": 0.2,
                    "min_len": 2,
                },
                "trainer_args": {
                    "batch_size": 32,
                    "effective_batch_size": 32,
                    "epochs": 5,
                    "info": True,
                    "sampler": None,
                    "gradient_clip": {
                        "clip_value": 1.0,
                    },
                    "shuffle": True,
                    "early_stopping": None,
                },
                "model": {
                    "linear": True,
                    "hidden_size": 96,
                    "num_hidden_layers": 3,
                    "num_attention_heads": 3,
                    "intermediate_size": 64,
                    "type_vocab_size": 240,
                    "plusplus": True,
                    "abspos_embeddings": False,
                },
                "optimizer": {
                    "lr": 5e-4,
                    "eps": 1e-6,
                },
                "scheduler": {
                    "_target_": "transformers.get_linear_schedule_with_warmup",
                    "num_warmup_epochs": 2,
                    "num_training_epochs": 3,
                },
                "metrics": {
                    "top1": {
                        "_target_": "corebehrt.evaluation.metrics.PrecisionAtK",
                        "topk": 1,
                    },
                    "top10": {
                        "_target_": "corebehrt.evaluation.metrics.PrecisionAtK",
                        "topk": 10,
                    },
                    "mlm_loss": {
                        "_target_": "corebehrt.evaluation.metrics.LossAccessor",
                        "loss_name": "loss",
                    },
                },
            }
        )

    def test_pretrain(self):
        ### Call pretrain script

        main_train(self.config_path)

        ### Validate generated files.

        # 1: Copy of data and pretrain configuration files should be created
        #    in the outcomes dir
        self.check_config(
            join(self.pretrain_dir, DATA_CFG), f"./tests/data/features/{DATA_CFG}"
        )
        self.check_config(
            join(self.pretrain_dir, PRETRAIN_CFG),
            f"./tests/data/pretrain/{PRETRAIN_CFG}",
        )

        # 2: Check existence of some files
        # 2.1: Final/best checkpoint
        self.check_last_checkpoint()
        # 2.2: Other files
        for file_name in [
            "pids_train",
            "pids_val",
        ]:
            self.assertTrue(exists(join(self.pretrain_dir, f"{file_name}.pt")))
        self.assertTrue(
            exists(join(self.pretrain_dir, PROCESSED_DATA_DIR, "vocabulary.pt"))
        )

    def test_pretrain_with_existing_model(self):
        ### Call pretrain script to train initial model

        main_train(self.config_path)
        checkpoint_mtime = self.check_last_checkpoint()

        ### Call pretrain script again, check that checkpoint is updated

        main_train(self.config_path)
        self.assertNotEqual(self.check_last_checkpoint(), checkpoint_mtime)

    def test_pretrain_with_restart_model(self):
        ### Call pretrain script to train initial model

        main_train(self.config_path)
        checkpoint_mtime = self.check_last_checkpoint()

        ### Modify config to use new restart_model and new model dir
        new_model_dir = join(self.tmp_dir, "new_pretrain")
        self.config["paths"] |= {
            "restart_model": self.pretrain_dir,
            "model": new_model_dir,
        }
        self.set_config(self.config)

        ### Call pretrain script again, check that new model is created and that
        ### old checkpoint is not updated

        main_train(self.config_path)

        # 1: New model checkpoint
        self.check_last_checkpoint(new_model_dir)

        # 2: Old model checkpoint must not have been updated
        self.assertEqual(self.check_last_checkpoint(), checkpoint_mtime)

    def check_last_checkpoint(self, model_path: str = None) -> float:
        model_path = model_path or self.pretrain_dir
        cp_path = join(model_path, "checkpoints", "checkpoint_epoch999_end.pt")

        self.assertTrue(exists(cp_path))
        return getmtime(cp_path)
