"""TODO
  [ ] load module without optimizer_init
"""
from typing import Any, Optional
from functools import cached_property
import torch
from torch import nn, Tensor 
from lightning.pytorch.core.module import LightningModule
from tensordict import TensorDict
from torchmetrics import Metric, MetricCollection
from lightning.pytorch.cli import instantiate_class

class BaseModelModule(LightningModule):
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 metrics_init: Optional[dict[str, list | dict]] = None,
                 optimizer_init: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
#        self.save_hyperparameters()
        self.model = model.to_tensor_dict_module()
        self.criterion = criterion.to_tensor_dict_module()
        self.metrics_init = metrics_init
        self.optimizer_init = optimizer_init
        self.lr = optimizer_init['init_args']['lr'] # FIXME 

    @cached_property
    def val_metrics(self):
        return self.build_metrics(prefix='val_')

    @cached_property
    def test_metrics(self):
        return self.build_metrics(prefix='test_')

    def training_step(self, input: TensorDict) -> Tensor:
        output = self.model(input)
        loss = self.criterion(output)['loss']

        self.log('train_loss', loss, prog_bar=True)
        return loss

    @torch.inference_mode()
    def _eval_step(self,
                   stage: str,
                   input: TensorDict,
                   metrics: MetricCollection | None,
    ) -> None:
        output = self.model(input)
        loss = self.criterion(output)['loss']
        
        if metrics:
            metrics.update(
                preds=output['preds'],
                target=output['target'],
            )

        self.log(f'{stage}_loss', loss, prog_bar=True)

    def _on_eval_epoch_end(self,
                           stage: str,
                           metrics: MetricCollection | None,
    ) -> None:
        if metrics:
            log_dict = metrics.compute()
            self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, # type: ignore
                        input: TensorDict,
    ) -> None:
        if self.trainer.state.fn == "tuning":
            return
        return self._eval_step(stage='val', input=input, metrics=self.val_metrics)

    def on_validation_epoch_end(self):
        return self._on_eval_epoch_end(stage='val', metrics=self.val_metrics)

    def test_step(self, # type: ignore
                  input: TensorDict,
    ) -> None:
        return self._eval_step(stage='test', input=input, metrics=self.test_metrics)

    def on_test_epoch_end(self):
        return self._on_eval_epoch_end(stage='test', metrics=self.test_metrics)

    def build_metrics(self, prefix: Optional[str] = None) -> MetricCollection | None:
        if self.metrics_init is None: return None

        metrics: dict[str, Metric] = {}
        hit_threshold_list = self.metrics_init['hit_threshold_list']
        metrics_init = self.metrics_init['metrics']

        for hit_threshold in hit_threshold_list:
            suffix = f'{hit_threshold:.2f}'.replace('.', 'p')
            for key, value in metrics_init.items():
                metrics |= {
                    f'{key}_{suffix}': instantiate_class(args=hit_threshold, init=value),
                }

        metric_collection = MetricCollection(
            metrics=metrics,
            compute_groups=None,
            prefix=prefix,
        ).to(self.device)
        return metric_collection

# https://github.com/Lightning-AI/pytorch-lightning/issues/15340
    def configure_optimizers(self):
        self.optimizer_init['init_args']['lr'] = self.lr  # for learning rate finder
        self.optimizer = instantiate_class(
            args=self.model.parameters(), 
            init=self.optimizer_init,
        )

#        if self.scheduler_init:
#            scheduler = instantiate_class(args=self.optimizer, init=self.scheduler_init)
#            return (
#                [self.optimizer],
#                [f"scheduler": scheduler, **self.scheduler_init.get("lightning_args", {3)}],
#            )
        return self.optimizer

    def load_model_state_dict(self, state_dict):
        prefix = 'model.'
        model_state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}
        self.model.load_state_dict(model_state_dict)

    def load_model_ckpt(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=self.device)['state_dict']
        self.load_model_state_dict(state_dict)
