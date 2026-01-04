import torch
from me0.lightning.modelmodule import BaseModelModule
from tensordict import TensorDict
from torchmetrics import MetricCollection

class ModelModule(BaseModelModule):

    @torch.inference_mode()
    def _eval_step(self,
                  stage: str,
                  input: TensorDict,
                  metrics: MetricCollection,
    ) -> None:
        output = self.model(input)
        loss = self.criterion(output)['loss']

        metrics.update(output)
        self.log(f'{stage}_loss', loss, prog_bar=True)
