from typing import Any
from functools import cached_property
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from tensordict import TensorDict
from me0.data.datasets.base import TensorDictListDataset
from lightning.pytorch.cli import instantiate_class
import copy

class DataModule(LightningDataModule):
    train_set: TensorDictListDataset
    val_set: TensorDictListDataset
    test_set: TensorDictListDataset

    def __init__(self,
                 dataset_init: dict[str, Any],
                 train_dataset_init_args: dict[str, Any],
                 eval_dataset_init_args: dict[str, Any],
                 files = dict[str, str],
                 batch_size: int = 256,
                 eval_batch_size: int = 512,
    ) -> None:
        super().__init__()
        self.files = files
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size

        self.train_dataset_init = copy.deepcopy(dataset_init)
        self.train_dataset_init['init_args'] |= train_dataset_init_args 

        self.eval_dataset_init = copy.deepcopy(dataset_init)
        self.eval_dataset_init['init_args'] |= eval_dataset_init_args 

    @cached_property
    def train_set(self):
        return instantiate_class(args=self.files['train'], init=self.train_dataset_init)

    @cached_property
    def val_set(self):
        return instantiate_class(args=self.files['val'], init=self.eval_dataset_init)

    @cached_property
    def test_set(self):
        return instantiate_class(args=self.files['test'], init=self.eval_dataset_init)

    @cached_property
    def predict_set(self):
        return instantiate_class(args=self.files['predict'], init=self.eval_dataset_init)

    def teardown(self, stage:str) -> None:
        print(f'{stage=}')
        if stage == 'fit':
            delattr(self, 'train_set')
#            delattr(self, 'val_set')
        elif stage == 'test':
            delattr(self, 'test_set')
        elif stage == 'predict':
            delattr(self, 'predict_set')

    def train_dataloader(self):
        dataset = self.train_set
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=dataset.collate,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        )

    def _eval_dataloader(self, dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=dataset.collate,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return self._eval_dataloader(self.val_set)

    def test_dataloader(self):
        return self._eval_dataloader(self.test_set)

    def predict_dataloader(self):
        return self._eval_dataloader(self.predict_set)
