import abc
from typing import Any, cast
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.compute import _safe_divide
from tensordict import TensorDict

import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from hist.intervals import clopper_pearson_interval

import numpy as np
import scipy as sp
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.csgraph import connected_components

class SegmentMatchingGraph(Metric, metaclass=abc.ABCMeta):
    """input and target is supposed to be image with the shape of
    (batch_size, layer, ieta, strip)
    """
    edges: Tensor
    gen_muons: Tensor # the number of generated muons
    rec_segments_matched: Tensor # the number of reconsturcted segments matched with muons
    rec_segments_unmatched: Tensor # the number of reconsturcted segments not matched with muons

    is_differentiable = False
    higher_is_better = None
    full_state_update = True

    def __init__(self,
                 edges: Any,
                 hit_threshold: float,
                 min_hit_ratio: float,
                 min_matched_layers: int,
                 max_delta_strip: int,
                 max_delta_layer: int,
                 max_delta_ieta: int,
                 value: str,
                 **kwargs: Any,
    ) -> None:
        """
        """
        super().__init__(**kwargs)

        if not isinstance(edges, Tensor):
            edges = torch.Tensor(edges, dtype=torch.float)

        self.register_buffer(name='edges',
                             tensor=edges.clone().detach().type(torch.float))

        nbins = len(self.edges) - 1

        # the number of generated muons
        self.add_state(name="gen_muons",
                       default=torch.zeros(nbins, dtype=torch.long),
                       dist_reduce_fx="sum")

        # the number of reconstucted segments that are matched with generated muons
        self.add_state(name="rec_segments_matched",
                       default=torch.zeros(nbins, dtype=torch.long),
                       dist_reduce_fx="sum")

        # the number of reconstucted segments that are not matched with generated muons
        self.add_state("rec_segments_unmatched",
                       default=torch.zeros(nbins, dtype=torch.long),
                       dist_reduce_fx="sum")

        self.hit_threshold = hit_threshold
        self.min_hit_ratio = min_hit_ratio
        self.min_matched_layers = min_matched_layers
        self.max_delta_layer = max_delta_layer
        self.max_delta_ieta = max_delta_ieta
        self.max_delta_strip = max_delta_strip
        self.value = value

        if value == 'nlayers':
            self.get_value = self.compute_seg_nlayers
        elif value == 'ieta':
            self.get_value = self.compute_seg_ieta
        else:
            self.get_value = lambda x: None

    @property
    def last_bin(self) -> int:
        return len(self.edges) - 2

    def is_segment(self, input: Tensor) -> Tensor:
        return input[:,1].unique().size(0) >= self.min_matched_layers

    def _update(self, hits: Tensor, label: Tensor, target: Tensor, value: Tensor) -> None:
        ...

    def update(self, input: TensorDict) -> None:
        preds_batch = input.get('preds').gt(self.hit_threshold)
        target_batch = input.get('target').gt(0).long()
        value_batch = input.get(self.value)
        if value_batch is None:  # FIXME
            value_batch = input.get('pt')

        if 'indices' in input:
            indices_batch = input.get('indices')
            for preds, target, indices, value in zip(preds_batch, target_batch, indices_batch, value_batch):
                hits = indices[preds]
                label = target[preds]
                new_hits, inverse, counts = hits.unique(dim=0, return_inverse=True, return_counts=True)
                if counts.gt(1).any():
                    new_label = torch.zeros_like(counts)
                    for i in range(new_label.size(0)):
                        new_label[i] = label[inverse == i].max()
                    hits = new_hits
                    label = new_label
                self._update(hits, label, indices[target.bool()], value)
        else:
            for preds, target, value in zip(preds_batch, target_batch, value_batch):
                hits = preds.nonzero()
                label = target[preds]
                self._update(hits, label, target.nonzero(), value)

    def compute_seg_nhits(self, input: Tensor) -> Tensor:
        return input.size(0)

    def compute_nseg(self, input: Tensor) -> Tensor:
        return input.size(0)

    def compute_seg_nlayers(self, input: Tensor) -> Tensor:
        return input[:,1].unique().size(0)

    def compute_seg_ieta(self, input: Tensor) -> Tensor:
        return input[:,0].unique().max().item() + 1

    @property
    @abc.abstractmethod
    def numerator(self) -> Tensor:
        ...

    @property
    @abc.abstractmethod
    def denominator(self) -> Tensor:
        ...

    @staticmethod
    def compute(num: Tensor, denom: Tensor, reduce: bool = True) -> Tensor:
        # FIXME
        if reduce:
            num = num.sum()
            denom = denom.sum()

        return _safe_divide(num=num, denom=denom)

    def compute_error(self,
                      y: Tensor,
                      num: Tensor,
                      denom: Tensor,
                      coverage=0.68,
    ) -> tuple[Tensor, Tensor]:
        y = y.cpu().numpy()
        num = num.cpu().numpy()
        denom = denom.cpu().numpy()

        ylow, yup = clopper_pearson_interval(num, denom, coverage=coverage)

        yerr_low = y - ylow
        yerr_up = yup - y

        yerr_low = torch.from_numpy(yerr_low)
        yerr_up = torch.from_numpy(yerr_up)
        return yerr_low, yerr_up

    @property
    def low_edges(self) -> Tensor:
        return self.edges[:-1]

    @property
    def up_edges(self) -> Tensor:
        return self.edges[1:]

    @property
    def bin_centers(self) -> Tensor:
        return (self.low_edges + self.up_edges) / 2

    @property
    def bin_widths(self) -> Tensor:
        return self.up_edges - self.low_edges

    def plot(self, # type: ignore
             ax: Axes | None = None,
             xlabel: str | None = None,
             ylabel: str | None = None,
             label: str | None = None,
    ) -> tuple[Figure, Axes]:
        # FIXME
        num = self.numerator
        denom = self.denominator
        y = self.compute(num=num, denom=denom, reduce=False)

        if not isinstance(y, Tensor):
            raise TypeError(f"Expected y to be a single tensor but got {y}")
        y = y.cpu()
        yerr = [each.cpu() for each in self.compute_error(y, num, denom)]

        if ax is None:
            fig, ax = plt.subplots(constrained_layout=True)
        else:
            fig = ax.get_figure()
        fig = cast(Figure, fig)
        ax = cast(Axes, ax)

        x = self.bin_centers.cpu()
        # half width
        xerr = self.bin_widths.cpu() / 2

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        ax.errorbar(x=x, y=y, yerr=yerr, xerr=xerr, label=label)
        ax.set_ylim(0, 1)

        return fig, ax

    @property
    def rec_segments(self) -> Tensor:
        return self.rec_segments_matched + self.rec_segments_unmatched


class NSegmentsGraph(SegmentMatchingGraph):

    @property
    def numerator(self) -> Tensor:
        return self.rec_segments

    @property
    def denominator(self) -> Tensor:
        return self.rec_segments.sum() * torch.ones_like(self.rec_segments)

    def _update(self, hits: Tensor, label: Tensor, target: Tensor, value: Tensor) -> None:
        if hits.size(0) < self.min_matched_layers: 
            n_segments = 0

        else:
            ieta, layer, strip = hits.T
            delta_layer = torch.abs(layer[:, None] - layer[None, :])
            delta_ieta = torch.abs(ieta[:, None] - ieta[None, :])
            delta_strip = torch.abs(strip[:, None] - strip[None, :])
    
            is_linked =  (
                (delta_layer >= 1) & 
                (delta_layer <= self.max_delta_layer) &
                (delta_ieta <= self.max_delta_ieta) &
                (delta_strip <= self.max_delta_strip)
            )
    
            edge_index = torch.argwhere(is_linked).T.contiguous()
            adj = to_scipy_sparse_matrix(edge_index, num_nodes=layer.size(0))
            num_components, component = connected_components(adj)
            _, count_arr = np.unique(component, return_counts=True)
    
            n_segments = 0
            for comp_id in np.unique(component):
                mask = torch.from_numpy(component == comp_id)
                subset = hits[mask]
                if self.is_segment(subset):
                    n_segments += 1
        device = self.edges.device
        n_seg_tensor = torch.tensor([n_segments], dtype=torch.float, device=device)
        index = torch.bucketize(n_seg_tensor, self.edges, right=True) - 1
        index = index.clamp_(0, self.last_bin).item()

        self.rec_segments_matched[index] += 1

class SegmentEfficiencyGraph(SegmentMatchingGraph):

    @property
    def numerator(self) -> Tensor:
        return self.rec_segments_matched

    @property
    def denominator(self) -> Tensor:
        return self.gen_muons

    def _update(self, hits: Tensor, label: Tensor, target: Tensor, value: Tensor) -> None:
        if target.size(0) < self.min_matched_layers: return
        value = self.get_value(target) or value
        index = torch.bucketize(value, self.edges, right=True) - 1
        index = index.clamp_(0, self.last_bin)
        self.gen_muons[index] += int(self.is_segment(target))

        ieta, layer, strip = hits.T
        delta_layer = torch.abs(layer[:, None] - layer[None, :])
        delta_ieta = torch.abs(ieta[:, None] - ieta[None, :])
        delta_strip = torch.abs(strip[:, None] - strip[None, :])

        is_linked =  (
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
                    matched = True
        self.rec_segments_matched[index] += int(matched)

    def plot(self,
             ax: Axes | None = None,
             xlabel: str | None = None,
             ylabel: str | None = 'Efficiency',
             label: str | None = None,
    ) -> tuple[Figure, Axes]:
        return super().plot(
                ax=ax,
                xlabel=xlabel,
                ylabel=ylabel,
                label=label,
                )


class SegmentFakeRateGraph(SegmentMatchingGraph):

    def _update(self, hits: Tensor, label: Tensor, target: Tensor, value: Tensor) -> None:
        if hits.size(0) < self.min_matched_layers: return

        ieta, layer, strip = hits.T
        delta_layer = torch.abs(layer[:, None] - layer[None, :])
        delta_ieta = torch.abs(ieta[:, None] - ieta[None, :])
        delta_strip = torch.abs(strip[:, None] - strip[None, :])

        is_linked =  (
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
                    matched = True
                value = self.get_value(segment) or value
                index = torch.bucketize(value, self.edges, right=True) - 1
                index = index.clamp_(0, self.last_bin)
                self.rec_segments_matched[index] += int(is_matched)
                self.rec_segments_unmatched[index] += int(not is_matched)

    @property
    def numerator(self) -> Tensor:
        return self.rec_segments_unmatched

    @property
    def denominator(self) -> Tensor:
        return self.rec_segments

    def plot(self,
             ax: Axes | None = None,
             xlabel: str | None = None,
             ylabel: str | None = 'Fake Rate',
             label: str | None = None,
    ) -> tuple[Figure, Axes]:
        return super().plot(
                ax=ax,
                xlabel=xlabel,
                ylabel=ylabel,
                label=label,
                )
