from typing import Any, Tuple
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.compute import _safe_divide
from tensordict import TensorDict

import numpy as np
import scipy as sp
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.csgraph import connected_components

class ClusteringSegmentMatching(Metric):
    """input and target is supposed to be image with the shape of
    (batch_size, layer, ieta, strip) or (batch_size, ieta, layer, strip)
    """
    gen_muons: Tensor # the number of generated muons
    rec_segments_matched: Tensor # the number of reconstructed segments matched with muons
    rec_segments_unmatched: Tensor # the number of reconstructed segments not matched with muons

    is_differentiable = False
    higher_is_better = None
    full_state_update = True

    def __init__(self,
                 hit_threshold: float,
                 min_hit_ratio: float,
                 min_matched_layers: int,
                 max_delta_strip: int,
                 max_delta_layer: int,
                 max_delta_ieta: int,
                 **kwargs: Any,
    ) -> None:
        """
        """
        super().__init__(**kwargs)

        # the number of generated muons
        self.add_state(name="gen_muons",
                       default=torch.zeros(1, dtype=torch.long),
                       dist_reduce_fx="sum")

        # the number of reconstructed segments that are matched with generated muons
        self.add_state(name="rec_segments_matched",
                       default=torch.zeros(1, dtype=torch.long),
                       dist_reduce_fx="sum")

        # the number of reconstructed segments that are not matched with generated muons
        self.add_state("rec_segments_unmatched",
                       default=torch.zeros(1, dtype=torch.long),
                       dist_reduce_fx="sum")

        # the hit efficiency of reconstructed segments
        self.add_state(name="hit_eff",
                       default=torch.zeros(1, dtype=torch.float),
                       dist_reduce_fx="sum")

        # the fake hit rate of reconstructed segments
        self.add_state(name="fake_hit_rate",
                       default=torch.zeros(1, dtype=torch.float),
                       dist_reduce_fx="sum")

        self.hit_threshold = hit_threshold
        self.min_hit_ratio = min_hit_ratio
        self.min_matched_layers = min_matched_layers
        self.max_delta_layer = max_delta_layer
        self.max_delta_ieta = max_delta_ieta
        self.max_delta_strip = max_delta_strip

    def is_segment(self, input: Tensor) -> Tensor:
        """
        Args:
            input:
        """
        # count layer
        return input[:,1].unique().size(0) >= self.min_matched_layers


    def _update(self, hits: Tensor, label: Tensor) -> None:
        ieta, layer, strip = hits.T
        delta_layer = torch.abs(layer[:, None] - layer[None, :])
        delta_ieta = torch.abs(ieta[:, None] - ieta[None, :])
        delta_strip = torch.abs(strip[:, None] - strip[None, :])
        
        is_linked = (
            (delta_layer >= 1) & 
            (delta_layer <= self.max_delta_layer) & 
            (delta_ieta <= self.max_delta_ieta) & 
            (delta_strip <= self.max_delta_strip)
        )
        
        edge_index = torch.argwhere(is_linked).T.contiguous()
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=layer.size(0))
        num_components, component = connected_components(adj)
        _, count_arr = np.unique(component, return_counts=True)
        
        matched = False
        for count in count_arr.argsort()[::-1]:
            subset_mask = torch.from_numpy(np.isin(component, count))
            segment = hits[subset_mask]
            if self.is_segment(segment):
                segment_label = label[subset_mask]
                passed_layer = self.is_segment(segment[segment_label.bool()])
                hit_eff = segment_label.float().mean().item()
                is_matched = (hit_eff >= self.min_hit_ratio) & passed_layer
                if is_matched and matched:
                    print("[ERROR] find segment again!")
                if is_matched:
                    self.rec_segments_matched += 1 
                    matched = True
                else:
                    self.rec_segments_unmatched += 1 

    def update(self, input: TensorDict) -> None:
        preds_batch = input.get('preds').gt(self.hit_threshold)
        target_batch = input.get('target').gt(0).long()
        if 'indices' in input:
            indices_batch = input.get('indices')
            has_muon = target_batch.any(dim=1)
            for preds, target, indices in zip(preds_batch, target_batch, indices_batch):
                if preds.sum() < self.min_matched_layers:
                    continue
                hits = indices[preds]
                label = target[preds]
                new_hits, inverse, counts = hits.unique(dim=0, return_inverse=True, return_counts=True)
                if counts.gt(1).any():
                    new_label = torch.zeros_like(counts)
                    for i in range(new_label.size(0)):
                        new_label[i] = label[inverse == i].max()
                    hits = new_hits
                    label = new_label
                self._update(hits, label)
        else:
            has_muon = target_batch.any(dim=(1,2,3))
            for preds, target in zip(preds_batch, target_batch):
                if preds.sum() < self.min_matched_layers:
                    continue
                hits = preds.nonzero()
                label = target[preds]
                self._update(hits, label)
        self.gen_muons += has_muon.sum()

    @property
    def rec_segments(self) -> Tensor:
        return self.rec_segments_matched + self.rec_segments_unmatched


class SegmentHitEfficiency(ClusteringSegmentMatching):

    def compute(self) -> Tensor:
        return _safe_divide(num=self.hit_eff,
                            denom=self.rec_segments_matched)


class SegmentFakeHitRate(ClusteringSegmentMatching):

    def compute(self) -> Tensor:
        """Compute confusion matrix."""
        return _safe_divide(num=self.fake_hit_rate,
                            denom=self.rec_segments_matched)


class SegmentEfficiency(ClusteringSegmentMatching):

    def compute(self) -> Tensor:
        return _safe_divide(num=self.rec_segments_matched,
                            denom=self.gen_muons)


class SegmentFakeRate(ClusteringSegmentMatching):

    def compute(self) -> Tensor:
        """Compute confusion matrix."""
        return _safe_divide(num=self.rec_segments_unmatched,
                            denom=self.rec_segments)
