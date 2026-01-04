#!/usr/bin/env python
from me0.lightning.datamodule import DataModule
from me0.lightning.cli import MyLightningCLI
import mplhep as mh
import torch

def run(cli: MyLightningCLI):
    mh.style.use(mh.styles.CMS)
    cli.fit()
    cli.test('best')

def main():
    # FIXME
    torch.set_num_threads(2)

    cli = MyLightningCLI(
        datamodule_class=DataModule,
        seed_everything_default=1234,
        run=False, # used to de-activate automatic fitting.
        trainer_defaults={
            'max_epochs': 2,
            'accelerator': 'gpu',
            'devices': [0],
            'enable_progress_bar': True,
        },
        save_config_kwargs={
            'overwrite': True
        },
        auto_configure_optimizers=True,
    )

    run(cli=cli)

if __name__ == '__main__':
    main()
