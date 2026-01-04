from me0.lightning.modelmodule import BaseModelModule
import torch
from pathlib import Path
from tensordict import TensorDict
from torchmetrics import MetricCollection

from me0.metrics.segmentgraph import SegmentEfficiencyGraph
from me0.metrics.segmentgraph import SegmentFakeRateGraph
from me0.metrics.segmentgraph import SegmentHitsGraph

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class ModelModule(BaseModelModule):

    @torch.inference_mode()
    def _eval_step(self, # type: ignore
                  stage: str,
                  input: TensorDict,
                  metrics: MetricCollection,
    ) -> None:
        output = self.model(input)
        loss = self.criterion(output)['loss']

        metrics.update(output)

        if stage == 'test':
            self.test_eff_pt.update(
                preds=output['preds'],
                target=output['target'],
                values=output['pt'],
            )

            self.test_fake_rate_nhits.update(
                preds=output['preds'],
                target=output['target'],
                values=self.test_fake_rate_nhits.compute_seg_nhits(output['preds']),
            )

            self.test_nhits.update(
                preds=output['preds'],
                target=output['target'],
                values=self.test_nhits.compute_seg_nhits(output['preds']),
            )

        self.log(f'{stage}_loss', loss, prog_bar=True)

    def on_test_start(self):
        self.test_eff_pt = SegmentEfficiencyGraph(
            edges=torch.arange(10, 50, 1),
            hit_threshold=0.5,
            min_hit_ratio=0.6,
            min_matched_layers=4
        ).to(self.device)

        self.test_fake_rate_nhits = SegmentFakeRateGraph(
            edges=torch.arange(4, 20, 1),
            hit_threshold=0.5,
            min_hit_ratio=0.6,
            min_matched_layers=4
        ).to(self.device)

        self.test_nhits = SegmentHitsGraph(
            edges=torch.arange(0, 20, 1),
            hit_threshold=0.5,
            min_hit_ratio=0.6,
            min_matched_layers=4,
            compute_with_cache=False,
        ).to(self.device)

    def _save_fig(self,
                  fig: Figure,
                  output_name: str,
                  suffix_list: list[str] = ['.png', '.pdf'],

    ) -> None:
        log_dir = Path(self.trainer.log_dir) # type: ignore
        output_path = log_dir / output_name 

        for suffix in suffix_list:
            fig.savefig(output_path.with_suffix(suffix))
            plt.close(fig)

    def _on_eval_epoch_end(self,
                           stage: str,
                           metrics: MetricCollection
    ) -> None:
        log_dict = metrics.compute()
        self.log_dict(log_dict, prog_bar=True)

        if stage == 'test':
            fig, _ = self.test_eff_pt.plot(xlabel=r'Generated muon $p_{T}$ [GeV]')
            self._save_fig(fig, 'efficiency_pt')

            fig, _ = self.test_fake_rate_nhits.plot(xlabel=f'Number of hits in a segment')
            self._save_fig(fig, 'fake-rate_nhits')

            fig, _ = self.test_nhits.plot(xlabel=r'Number of predicted muon hits')
            self._save_fig(fig, 'nhits')
