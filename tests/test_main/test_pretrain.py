# from os.path import exists, getmtime, join

# import yaml

# from corebehrt.constants.paths import DATA_CFG, PRETRAIN_CFG
# from corebehrt.main.pretrain import main_train

# from .base import TestMainScript


# class TestCreateOutcomes(TestMainScript):
#     def setUp(self):
#         super().setUp()

#         self.pretrain_dir = join(self.tmp_dir, "pretrain")

#         # Load config from pretrain config file
#         pretrain_config_path = f"./tests/data/pretrain/{PRETRAIN_CFG}"
#         with open(pretrain_config_path) as f:
#             config = yaml.safe_load(f)

#         # Update paths in the loaded config
#         config["paths"] = {
#             "prepared_data": "./tests/data/pretrain/processed_data",
#             "tokenized": "./tests/data/tokenized",
#             "model": self.pretrain_dir,
#         }

#         self.set_config(config)

#     def test_pretrain(self):
#         ### Call pretrain script

#         main_train(self.config_path)

#         ### Validate generated files.

#         # 1: Copy of data and pretrain configuration files should be created
#         #    in the outcomes dir
#         self.check_config(
#             join(self.pretrain_dir, DATA_CFG), f"./tests/data/features/{DATA_CFG}"
#         )

#         # 2: Check existence of some files
#         # 2.1: Final/best checkpoint
#         self.check_last_checkpoint()
  
#     def test_pretrain_with_existing_model(self):
#         ### Call pretrain script to train initial model

#         main_train(self.config_path)
#         checkpoint_mtime = self.check_last_checkpoint()

#         ### Call pretrain script again, check that checkpoint is updated

#         main_train(self.config_path)
#         self.assertNotEqual(self.check_last_checkpoint(), checkpoint_mtime)

#     def test_pretrain_with_restart_model(self):
#         ### Call pretrain script to train initial model

#         main_train(self.config_path)
#         checkpoint_mtime = self.check_last_checkpoint()

#         ### Modify config to use new restart_model and new model dir
#         new_model_dir = join(self.tmp_dir, "new_pretrain")
#         self.config["paths"] |= {
#             "restart_model": self.pretrain_dir,
#             "model": new_model_dir,
#         }
#         self.set_config(self.config)

#         ### Call pretrain script again, check that new model is created and that
#         ### old checkpoint is not updated

#         main_train(self.config_path)

#         # 1: New model checkpoint
#         self.check_last_checkpoint(new_model_dir)

#         # 2: Old model checkpoint must not have been updated
#         self.assertEqual(self.check_last_checkpoint(), checkpoint_mtime)

#     def check_last_checkpoint(self, model_path: str = None) -> float:
#         model_path = model_path or self.pretrain_dir
#         cp_path = join(model_path, "checkpoints", "checkpoint_epoch999_end.pt")

#         self.assertTrue(exists(cp_path))
#         return getmtime(cp_path)
