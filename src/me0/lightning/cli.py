import torch
from pathlib import Path
from torch import Tensor
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.cli import LightningCLI

from me0.utils.learningcurve import make_learning_curves

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--ckpt_path", type=str, help="Path to checkpoint for resuming training")
        parser.add_argument("--log_dir", type=str, help="Path to save logs")

    def instantiate_trainer(self, **kwargs):
        log_dir = self.config.get('log_dir', None)

        if log_dir:
            logger = CSVLogger(save_dir=log_dir, name=None)
            kwargs['logger'] = logger
        
        return super().instantiate_trainer(**kwargs)

    def fit(self):
        ckpt_path = self.config.get('ckpt_path', None)
        self.trainer.validate(self.model, datamodule=self.datamodule, ckpt_path=ckpt_path)

        if ckpt_path:
            self.trainer.fit(self.model, datamodule=self.datamodule, ckpt_path=ckpt_path)
        else:
            tuner = Tuner(self.trainer)
            # LR-Finder 
            lr_finder = tuner.lr_find(
                model=self.model, 
                datamodule=self.datamodule
            )
            fig = lr_finder.plot(suggest=True)
            fig.savefig(f'{self.trainer.log_dir}/lr_finder.png')
            new_lr = lr_finder.suggestion()
            self.model.lr = new_lr
            self.trainer.fit(self.model, datamodule=self.datamodule)

    def test(self, ckpt_path: str | Path | None = None):
        self.trainer.test(ckpt_path=ckpt_path, datamodule=self.datamodule)

        if ckpt_path == 'best':
            make_learning_curves(log_dir=Path(self.trainer.log_dir))

    def load_model_ckpt(self, ckpt_path: str | Path | None = None):
        ckpt_path = ckpt_path or self.config.get('ckpt_path', None)

        if ckpt_path is None:
            raise TypeError(
                "`load_model_ckpt` requires a `ckpt_path`, either provided directly or set in the config."
            )
        self.model.load_model_ckpt(ckpt_path)
