import abc
from typing import Any, Optional, cast
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.compute import _safe_divide
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from hist.intervals import clopper_pearson_interval


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
                 **kwargs: Any,
    ) -> None:
        """
        """
        super().__init__(**kwargs)

        if not isinstance(edges, Tensor):
            edges = torch.Tensor(edges, dtype=torch.float)

        self.register_buffer(name='edges',
                             tensor=edges.clone().detach().type(torch.float))
#                             tensor=torch.tensor(edges, dtype=torch.float))

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

    @property
    def last_bin(self) -> int:
        return len(self.edges) - 2

    def has_segment(self, input: Tensor) -> Tensor:
        # pp mean predicted postiive
        # sum along (ieta, strip) -> output: (batch_size, layer)
        has_pp_per_layer = input.any(dim=(1, 3)) # FIXME

        # sum along layer
        num_layers_with_pp = has_pp_per_layer.sum(dim=-1)
        return num_layers_with_pp >= self.min_matched_layers

    def update(self, preds: Tensor, target: Tensor, values: Tensor) -> None:
        preds = preds.gt(self.hit_threshold)
        has_muon = target.gt(0).any(dim=(1, 2, 3))

        # predicted postivie = rec segment
        has_rec_seg = self.has_segment(preds)

        hit_is_true_pos = torch.logical_and(preds, target)
        passed_layers = self.has_segment(hit_is_true_pos) 
        passed_ratio = (hit_is_true_pos.sum(dim=(1, 2, 3)) / preds.sum(dim=(1, 2, 3))) >= self.min_hit_ratio

        is_matched = passed_layers & passed_ratio
        is_unmatched = ~is_matched

        has_rec_seg_matched = has_rec_seg & is_matched
        has_rec_seg_unmatched = has_rec_seg & is_unmatched

        index = torch.bucketize(values.reshape(-1), self.edges, right=True) - 1
        index = index.clamp_(0, self.last_bin)

        self.gen_muons.index_add_(dim=0, index=index, source=has_muon.long())
        self.rec_segments_matched.index_add_(dim=0, index=index, source=has_rec_seg_matched.long())
        self.rec_segments_unmatched.index_add_(dim=0, index=index, source=has_rec_seg_unmatched.long())

    def compute_seg_nhits(self, preds: Tensor) -> Tensor:
        batch_size = preds.size(0)
        return preds.gt(self.hit_threshold).reshape(batch_size, -1).sum(dim=1)

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
             num: Tensor | None = None,
             denom: Tensor | None = None,
             ax: Axes | None = None,
             xlabel: str | None = None,
             ylabel: str | None = None,
             label: str | None = None,
    ) -> tuple[Figure, Axes]:
        # FIXME
        num = num if num != None else self.numerator
        denom = denom if denom != None else self.denominator
        
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


class SegmentEfficiencyGraph(SegmentMatchingGraph):

    @property
    def numerator(self) -> Tensor:
        return self.rec_segments_matched

    @property
    def denominator(self) -> Tensor:
        return self.gen_muons

    def plot(self,
             y: Tensor | None = None,
             ax: Axes | None = None,
             xlabel: str | None = None,
             ylabel: str | None = 'Efficiency',
    ) -> tuple[Figure, Axes]:
        return super().plot(ylabel=ylabel)


class SegmentFakeRateGraph(SegmentMatchingGraph):

    @property
    def numerator(self) -> Tensor:
        return self.rec_segments_unmatched

    @property
    def denominator(self) -> Tensor:
        return self.rec_segments_matched + self.rec_segments_unmatched

    def plot(self,
             y: Tensor | None = None,
             ax: Axes | None = None,
             xlabel: str | None = None,
             ylabel: str | None = 'Fake Rate',
    ) -> tuple[Figure, Axes]:
        return super().plot(ylabel=ylabel)


class SegmentHitsGraph(SegmentMatchingGraph):

    # FIXME
    @property
    def numerator(self) -> Tensor | None:
        return None 

    @property
    def denominator(self) -> Tensor | None:
        return None 

    def plot(self,
             y: Tensor | None = None,
             ax: Axes | None = None,
             xlabel: str | None = None,
             ylabel: str | None = 'Normalized'
    ) -> tuple[Figure, Axes]:
        fig, ax = super().plot(num=self.rec_segments_matched,
                               denom=self.rec_segments_matched.sum(),
                               ax=ax,
                               xlabel=xlabel,
                               ylabel=ylabel,
                               label='rec_segments_matched')

        fig, ax = super().plot(num=self.rec_segments_unmatched,
                               denom=self.rec_segments_unmatched.sum(),
                               ax=ax,
                               xlabel=xlabel,
                               ylabel=ylabel,
                               label='rec_segments_unmatched')
        ax.legend()
        return fig, ax 
