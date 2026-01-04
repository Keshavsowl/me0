import torch
from torch import nn, Tensor
from torch.nn.functional import binary_cross_entropy_with_logits
from typing import Optional, Union
from tensordict.nn import TensorDictModule

def masked_bce_loss(
        input: Tensor,
        target: Tensor,
        data_mask: Tensor,
        reduction: str = 'mean1',
        pos_weight: float | Tensor | None = None,
    ) -> Tensor:
    """
    Args:
    Returns:
    """
    if isinstance(pos_weight, (int, float)):
        pos_weight = torch.tensor(pos_weight, dtype=torch.float, device=input.device)

    target = target.gt(0).type(torch.float32) # FIXME
    loss = binary_cross_entropy_with_logits(input, target, reduction='none', pos_weight=pos_weight)
    loss.masked_fill_(~data_mask, 0)

    if reduction == 'mean0':
        loss = loss.mean()
    elif reduction == 'mean1':
        loss = loss.sum()/(data_mask).sum()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss


class ME0BCELoss(nn.Module):
    def __init__(
        self,
        pos_weight: float | Tensor,
        reduction: str = 'mean1',
    ) -> None:
        """
        Args:
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(
        self,
        input: Tensor, 
        target: Tensor, 
        data_mask: Tensor,
    ) -> Tensor:
        """
        Args:
        Returns:
        """
        return masked_bce_loss(input=input,
                               target=target,
                               data_mask=data_mask,
                               reduction=self.reduction,
                               pos_weight=self.pos_weight)

    def to_tensor_dict_module(self):
        return TensorDictModule(
            module=self,
            in_keys=['logits', 'target', 'data_mask'],
            out_keys=['loss'],
            inplace=False,
        )
