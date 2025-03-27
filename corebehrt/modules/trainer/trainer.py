import os
from collections import namedtuple

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from corebehrt import azure
from corebehrt.functional.trainer.collate import dynamic_padding
from corebehrt.functional.trainer.log import log_number_of_trainable_parameters
from corebehrt.modules.monitoring.logger import get_tqdm
from corebehrt.modules.monitoring.metric_aggregation import (
    compute_avg_metrics,
    save_curves,
    save_metrics_to_csv,
    save_predictions,
)
from corebehrt.modules.setup.config import Config, instantiate_class
from corebehrt.modules.trainer.freezing import freeze_bottom_layers, unfreeze_all_layers
from corebehrt.modules.trainer.utils import is_plateau

yaml.add_representer(Config, lambda dumper, data: data.yaml_repr(dumper))

BEST_MODEL_ID = 999  # For backwards compatibility
DEFAULT_CHECKPOINT_FREQUENCY = 100


class EHRTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset = None,
        test_dataset: Dataset = None,
        val_dataset: Dataset = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler.StepLR = None,
        metrics: dict = {},
        args: dict = {},
        sampler: callable = None,
        cfg=None,
        logger=None,
        accumulate_logits: bool = False,
        run_folder: str = None,
        last_epoch: int = None,
    ):
        self._initialize_basic_attributes(
            model,
            train_dataset,
            test_dataset,
            val_dataset,
            optimizer,
            scheduler,
            metrics,
            sampler,
            cfg,
            accumulate_logits,
            last_epoch,
        )
        self.logger = logger
        self._set_default_args(args)
        self.run_folder = run_folder or cfg.paths.model
        self.log("Initialize metrics")
        self.metrics = (
            {k: instantiate_class(v) for k, v in metrics.items()} if metrics else {}
        )
        self.scaler = torch.GradScaler(device=self.device.type)

        self._initialize_early_stopping()
        self._initialize_freezing()
        log_number_of_trainable_parameters(self.model)

    def _initialize_basic_attributes(
        self,
        model,
        train_dataset,
        test_dataset,
        val_dataset,
        optimizer,
        scheduler,
        metrics,
        sampler,
        cfg,
        accumulate_logits,
        last_epoch,
    ):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = (
            {k: instantiate_class(v) for k, v in metrics.items()} if metrics else {}
        )
        self.sampler = sampler
        self.cfg = cfg
        self.accumulate_logits = accumulate_logits
        self.continue_epoch = last_epoch + 1 if last_epoch is not None else 0

        self.already_unfrozen = False

    def _initialize_freezing(self):
        self.already_unfrozen = True
        if self.args.get("n_layers_to_freeze", 0) > 0:
            self.model = freeze_bottom_layers(
                self.model, self.args.get("n_layers_to_freeze", 0)
            )
            self.already_unfrozen = False

        self.unfreeze_on_plateau = self.args.get("unfreeze_on_plateau", False)
        self.unfreeze_at_epoch = self.args.get("unfreeze_at_epoch", None)

        if self.unfreeze_at_epoch is not None:
            self.log(f"Model will be unfrozen at epoch {self.unfreeze_at_epoch}")

    def _set_default_args(self, args):
        default_args = {
            "save_every_k_steps": float("inf"),
            "collate_fn": dynamic_padding,
        }
        self.args = {**default_args, **args}
        self.log(f"Trainer args: {self.args}")
        if not (self.args["effective_batch_size"] % self.args["batch_size"] == 0):
            raise ValueError("effective_batch_size must be a multiple of batch_size")

    def _initialize_early_stopping(self):
        early_stopping = self.args.get("early_stopping", False)
        self.early_stopping = True if early_stopping else False
        self.early_stopping_patience = (
            early_stopping if early_stopping else 1000
        )  # Set patience parameter, for example, to 10 epochs.
        self.early_stopping_counter = (
            0  # Counter to keep track of epochs since last best val loss
        )
        self.stop_training = False

        # Get the metric to use for early stopping from the config
        self.stopping_metric = self.args.get("stopping_criterion", "val_loss")

        # Check if the specified metric is available in our metrics
        metric_exists = (
            self.stopping_metric == "val_loss" or self.stopping_metric in self.metrics
        )
        if not metric_exists:
            self.log(
                f"WARNING: Specified stopping metric '{self.stopping_metric}' is not available in metrics. Falling back to 'val_loss'."
            )
            self.log("Available metrics:")
            for metric in self.metrics:
                self.log(f"- {metric}")
            self.stopping_metric = "val_loss"

        self.log(
            f"Early stopping: {self.early_stopping} with patience {self.early_stopping_patience} using metric '{self.stopping_metric}'"
        )
        self.best_metric_value = None

    def train(self, **kwargs):
        self.log(f"Torch version {torch.__version__}")
        self._update_attributes(**kwargs)

        self.accumulation_steps: int = (
            self.args["effective_batch_size"] // self.args["batch_size"]
        )
        dataloader = self.setup_training()
        self.log("Test validation before starting training")
        self.validate_and_log(0, [0], dataloader)
        for epoch in range(self.continue_epoch, self.args["epochs"]):
            self._train_epoch(epoch, dataloader)
            if self.stop_training:
                break

    def _train_epoch(self, epoch: int, dataloader: DataLoader) -> None:
        if self._should_unfreeze_at_epoch(epoch):
            self._unfreeze_model(f"Reached epoch {epoch}!")

        train_loop = get_tqdm(dataloader)
        train_loop.set_description(f"Train {epoch}")
        epoch_loss = []
        step_loss = 0
        metrics = []
        for i, batch in enumerate(train_loop):
            step_loss += self._train_step(batch).item()
            if (i + 1) % self.accumulation_steps == 0:
                self._clip_gradients()
                self._update()
                self._accumulate_metrics(
                    metrics, step_loss, epoch_loss, step=(epoch * len(train_loop)) + i
                )
                step_loss = 0
        self._log_batch(metrics)
        self.validate_and_log(epoch, epoch_loss, train_loop)
        torch.cuda.empty_cache()
        del train_loop
        del epoch_loss

    def _clip_gradients(self):
        # Then clip them if needed
        if self.args.get("gradient_clip", False):
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.args.get("gradient_clip", {}).get("max_norm", 1.0),
            )

    def _train_step(self, batch: dict):
        self.optimizer.zero_grad()
        self.batch_to_device(batch)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            loss = self.model(batch).loss
        self.scaler.scale(loss).backward()

        return loss

    def _update(self):
        """Updates the model (optimizer and scheduler)"""
        """Updates the model and logs the loss"""
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()

    def _accumulate_metrics(self, metrics, step_loss, epoch_loss, step):
        """Accumulates the metrics"""

        epoch_loss.append(step_loss / self.accumulation_steps)
        metrics.append(
            azure.metric("Train loss", step_loss / self.accumulation_steps, step)
        )

        if self.args["info"]:
            for param_group in self.optimizer.param_groups:
                current_lr = param_group["lr"]
                metrics.append(azure.metric("Learning Rate", current_lr, step))
                break

    def validate_and_log(
        self, epoch: int, epoch_loss: float, train_loop: DataLoader
    ) -> None:
        val_loss, val_metrics = self._evaluate(epoch, mode="val")
        _, test_metrics = self._evaluate(epoch, mode="test")
        if epoch == 1:  # for testing purposes/if first epoch is best
            self._save_checkpoint(
                epoch,
                train_loss=epoch_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                final_step_loss=epoch_loss[-1],
                best_model=True,
            )

        self._self_log_results(
            epoch, val_loss, val_metrics, epoch_loss, len(train_loop)
        )

        current_metric_value = val_metrics.get(
            self.stopping_metric, val_loss
        )  # get the metric we monitor. Same as early stopping

        if self._should_unfreeze_on_plateau(current_metric_value):
            self._unfreeze_model("Performance plateau detected!")

        if self._should_stop_early(
            epoch, current_metric_value, val_loss, epoch_loss, val_metrics, test_metrics
        ):
            return
        self._save_checkpoint_conditionally(
            epoch, epoch_loss, val_loss, val_metrics, test_metrics
        )

    def _save_checkpoint_conditionally(
        self,
        epoch: int,
        epoch_loss: float,
        val_loss: float,
        val_metrics,
        test_metrics: dict,
    ) -> None:
        should_save = (
            epoch % self.args.get("checkpoint_frequency", DEFAULT_CHECKPOINT_FREQUENCY)
            == 0
        )
        if should_save:
            self._save_checkpoint(
                epoch,
                train_loss=epoch_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                final_step_loss=epoch_loss[-1],
                best_model=True,
            )

    def _self_log_results(
        self,
        epoch: int,
        val_loss: float,
        val_metrics: dict,
        epoch_loss: float,
        len_train_loop: int,
    ) -> None:
        for k, v in val_metrics.items():
            self.run_log(name=k, value=v, step=epoch)
        self.run_log(name="Val loss", value=val_loss, step=epoch)
        self.log(
            f"Epoch {epoch} train loss: {sum(epoch_loss) / (len_train_loop / self.accumulation_steps)}"
        )
        self.log(f"Epoch {epoch} val loss: {val_loss}")
        self.log(f"Epoch {epoch} metrics: {val_metrics}\n")

    def _should_stop_early(
        self,
        epoch,
        current_metric_value: float,
        val_loss: float,
        epoch_loss: float,
        val_metrics: dict,
        test_metrics: dict = {},
    ) -> bool:
        if not self.early_stopping:
            return False
        # Get the current value of the metric
        self._initialize_best_metric_value(current_metric_value)
        if self._is_improvement(current_metric_value):
            self.best_metric_value = current_metric_value
            self.early_stopping_counter = 0
            self._save_checkpoint(
                epoch,
                train_loss=epoch_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                final_step_loss=epoch_loss[-1],
                best_model=True,
            )
            return False
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_patience:
                self.log("Early stopping triggered!")
                self.stop_training = True
                return True
        return False

    def _is_improvement(self, current_metric_value):
        """Returns True if the current metric value is an improvement over the best metric value"""
        if self.best_metric_value is None:
            return True
        if self.stopping_metric == "val_loss":
            return current_metric_value < self.best_metric_value
        else:
            return current_metric_value > self.best_metric_value

    def _initialize_best_metric_value(self, current_metric_value: float) -> None:
        if not hasattr(self, "best_metric_value"):
            self.best_metric_value = current_metric_value

    def setup_training(self) -> DataLoader:
        """Sets up the training dataloader and returns it"""
        self.model.train()
        self.save_setup()
        dataloader = self.get_dataloader(self.train_dataset, mode="train")
        return dataloader

    def _evaluate(self, epoch: int, mode="val") -> tuple:
        """Returns the validation/test loss and metrics"""
        if mode == "val":
            if self.val_dataset is None:
                self.log("No validation dataset provided")
                return None, None
            dataloader = self.get_dataloader(self.val_dataset, mode="val")
        elif mode == "test":
            if self.test_dataset is None:
                self.log("No test dataset provided")
                return None, None
            dataloader = self.get_dataloader(self.test_dataset, mode="test")
        else:
            raise ValueError(f"Mode {mode} not supported. Use 'val' or 'test'")

        self.model.eval()
        loop = get_tqdm(dataloader)
        loop.set_description(mode)
        loss = 0

        metric_values = {name: [] for name in self.metrics}
        logits_list = [] if self.accumulate_logits else None
        targets_list = [] if self.accumulate_logits else None

        with torch.no_grad():
            for batch in loop:
                self.batch_to_device(batch)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = self.model(batch)
                loss += outputs.loss.item()

                if self.accumulate_logits:
                    logits_list.append(
                        outputs.logits.float().cpu()
                    )  # .float to convert to float32 (from bfloat16)
                    targets_list.append(batch["target"].cpu())
                else:
                    for name, func in self.metrics.items():
                        metric_values[name].append(func(outputs, batch))

        if self.accumulate_logits:
            metric_values = self.process_binary_classification_results(
                logits_list, targets_list, epoch, mode=mode
            )
        else:
            metric_values = compute_avg_metrics(metric_values)

        self.model.train()

        return loss / len(loop), metric_values

    def process_binary_classification_results(
        self, logits: list, targets: list, epoch: int, mode="val"
    ) -> dict:
        """Process results specifically for binary classification."""
        targets = torch.cat(targets)
        logits = torch.cat(logits)
        batch = {"target": targets}
        outputs = namedtuple("Outputs", ["logits"])(logits)
        metrics = {}
        for name, func in self.metrics.items():
            v = func(outputs, batch)
            self.log(f"{name}: {v}")
            metrics[name] = v
        save_curves(self.run_folder, logits, targets, epoch, mode)
        save_metrics_to_csv(self.run_folder, metrics, epoch, mode)
        save_curves(self.run_folder, logits, targets, BEST_MODEL_ID, mode)
        save_metrics_to_csv(
            self.run_folder, metrics, BEST_MODEL_ID, mode
        )  # For compatibility / best model
        save_predictions(self.run_folder, logits, targets, BEST_MODEL_ID, mode)
        return metrics

    def get_dataloader(self, dataset, mode) -> DataLoader:
        """Returns a dataloader for the dataset"""
        if mode == "val":
            batchsize = self.args.get("val_batch_size", self.args["batch_size"])
        elif mode == "test":
            batchsize = self.args.get("test_batch_size", self.args["batch_size"])
        else:
            batchsize = self.args["batch_size"]
        return DataLoader(
            dataset,
            batch_size=batchsize,
            shuffle=False,
            collate_fn=self.args["collate_fn"],
        )

    def batch_to_device(self, batch: dict) -> None:
        """Moves a batch to the device in-place"""
        for key, value in batch.items():
            batch[key] = value.to(self.device)

    def log(self, message: str) -> None:
        """Logs a message to the logger and stdout"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def run_log(self, name, value, step=None):
        if azure.is_mlflow_available():
            azure.log_metric(name, value, step=step)
        else:
            self.log(f"{name}: {value}")

    def _log_batch(self, metrics: list):
        if azure.is_mlflow_available():
            azure.log_batch(metrics=metrics)
        else:
            self.log(metrics)

    def save_setup(self) -> None:
        """Saves the config and model config"""
        self.model.config.save_pretrained(self.run_folder)
        with open(os.path.join(self.run_folder, "pretrain_config.yaml"), "w") as file:
            yaml.dump(self.cfg.to_dict(), file)

    def _save_checkpoint(self, epoch: int, best_model=False, **kwargs) -> None:
        """Saves a checkpoint. Model with optimizer and scheduler if available."""
        # Model/training specific
        id = epoch if not best_model else BEST_MODEL_ID
        os.makedirs(os.path.join(self.run_folder, "checkpoints"), exist_ok=True)
        checkpoint_name = os.path.join(
            self.run_folder, "checkpoints", f"checkpoint_epoch{id}_end.pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                **kwargs,
            },
            checkpoint_name,
        )

    def _update_attributes(self, **kwargs):
        for key, value in kwargs.items():
            if key == "args":
                self.args = {**self.args, **value}
            else:
                setattr(self, key, value)

    def _should_unfreeze_at_epoch(self, epoch: int) -> bool:
        """Determine if we should unfreeze all layers based on current epoch."""
        return (
            self.unfreeze_at_epoch is not None
            and epoch >= self.unfreeze_at_epoch
            and not getattr(self, "already_unfrozen", False)
        )

    def _should_unfreeze_on_plateau(self, current_metric_value: float) -> bool:
        """Determine if we should unfreeze all layers based on plateau detection."""
        if (
            not self.unfreeze_on_plateau
            or getattr(self, "already_unfrozen", False)
            or self.best_metric_value is None
        ):
            return False

        return is_plateau(
            self.best_metric_value,
            current_metric_value,
            self.args.get("plateau_threshold", 0.01),
        )

    def _unfreeze_model(self, reason: str):
        """Unfreeze all layers and handle related state updates."""
        self.log(f"{reason} Unfreezing all layers of the model")
        self.model = unfreeze_all_layers(self.model)

        # Mark as unfrozen to avoid repeated unfreezing
        self.already_unfrozen = True

        # Optionally reset early stopping counter after unfreezing
        if self.args.get("reset_patience_after_unfreeze", True):
            self.early_stopping_counter = 0
            self.log("Reset early stopping counter after unfreezing")
