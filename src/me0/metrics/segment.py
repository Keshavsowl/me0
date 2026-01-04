from typing import Any, Tuple
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.compute import _safe_divide
from tensordict import TensorDict

class SegmentMatching(Metric):
    """input and target is supposed to be image with the shape of
    (batch_size, layer, ieta, strip) or (batch_size, ieta, layer, strip)
    """
    gen_muons: Tensor # the number of generated muons
    rec_segments_matched: Tensor # the number of reconsturcted segments matched with muons
    rec_segments_unmatched: Tensor # the number of reconsturcted segments not matched with muons

    is_differentiable = False
    higher_is_better = None
    full_state_update = True

    def __init__(self,
                 hit_threshold: float,
                 min_hit_ratio: float,
                 min_matched_layers: int,
#                 layer_first: bool,
                 **kwargs: Any,
    ) -> None:
        """
        """
        super().__init__(**kwargs)

        # the number of generated muons
        self.add_state(name="gen_muons",
                       default=torch.zeros(1, dtype=torch.long),
                       dist_reduce_fx="sum")

        # the number of reconstucted segments that are matched with generated muons
        self.add_state(name="rec_segments_matched",
                       default=torch.zeros(1, dtype=torch.long),
                       dist_reduce_fx="sum")

        # the number of reconstucted segments that are not matched with generated muons
        self.add_state("rec_segments_unmatched",
                       default=torch.zeros(1, dtype=torch.long),
                       dist_reduce_fx="sum")

        self.hit_threshold = hit_threshold
        self.min_hit_ratio = min_hit_ratio
        self.min_matched_layers = min_matched_layers

    def has_segment(self, input: Tensor) -> Tensor:
        # pp mean predicted postiive
        # sum along (ieta, strip) -> output: (batch_size, layer)
        has_pp_per_layer = input.any(dim=(1, 3))  # FIXME

        # sum along layer
        num_layers_with_pp = has_pp_per_layer.sum(dim=-1)
        return num_layers_with_pp >= self.min_matched_layers

    def _update(self, preds: Tensor, target: Tensor) -> None:
        preds = preds.gt(self.hit_threshold)
        has_muon = target.gt(0).any(dim=(1, 2, 3))

        # predicted postivie = rec segment
        has_rec_seg = self.has_segment(preds)

        hit_is_true_pos = torch.logical_and(preds, target)
        passed_layers = self.has_segment(hit_is_true_pos) 
        passed_ratio = (hit_is_true_pos.sum(dim=(1, 2, 3)) / preds.sum(dim=(1, 2, 3))) >= self.min_hit_ratio

        is_matched = passed_layers & passed_ratio
        is_unmatched = ~is_matched

        has_matched_rec_seg = has_rec_seg & is_matched
        has_unmatched_rec_seg = has_rec_seg & is_unmatched

        self.gen_muons = self.gen_muons + has_muon.sum()
        self.rec_segments_matched = self.rec_segments_matched + has_matched_rec_seg.sum()
        self.rec_segments_unmatched = self.rec_segments_unmatched + has_unmatched_rec_seg.sum()

    def update(self, input: TensorDict) -> None:
        self._update(input['preds'], input['target'])

    @property
    def rec_segments(self) -> Tensor:
        return self.rec_segments_matched + self.rec_segments_unmatched


class SegmentEfficiency(SegmentMatching):

    def compute(self) -> Tensor:
        return _safe_divide(num=self.rec_segments_matched,
                            denom=self.gen_muons)

class SegmentFakeRate(SegmentMatching):

    def compute(self) -> Tensor:
        """Compute confusion matrix."""
        return _safe_divide(num=self.rec_segments_unmatched,
                            denom=self.rec_segments)
