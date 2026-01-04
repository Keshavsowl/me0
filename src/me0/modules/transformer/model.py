from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
from tensordict.nn import TensorDictModule
import einops as eo

class TransformerEncoder(nn.TransformerEncoder):

    def __init__(
        self,
        num_layers: int,
        dim_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = 'gelu',
        layer_norm_eps: float = 0.00001,
        norm_first: bool = False,
        bias: bool = True,
    ) -> None:
        """
        """
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True, # NOTE:
            norm_first=norm_first,
            bias=bias,
        )

        super().__init__(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.num_heads = num_heads

    def forward( # type: ignore
        self,
        input: Tensor,
        input_pad_mask: Tensor | None = None,
    ) -> Tensor:
        """
        """
        if input_pad_mask is None:
            attn_mask = None
        else:
            attn_mask = eo.repeat(
                tensor=input_pad_mask,
                pattern='n s -> (n h) t s',
                h=self.num_heads,
                t=input_pad_mask.size(1),
            )

        return super().forward(
            src=input,
            mask=attn_mask,
            src_key_padding_mask=input_pad_mask,
            is_causal=False,
        )

class ME0Transformer(nn.Module):

    def __init__(
        self,
        dim_input: int,
        num_layers: int,
        dim_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        layer_norm_eps: float = 0.00001,
        norm_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(dim_input, dim_model)
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            bias=bias,
        )
        self.head = nn.Linear(dim_model, 1)

    def forward(self, x: Tensor, data_mask: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.embedding(x)
        x = self.encoder(x, data_mask.logical_not())
        x = self.head(x).squeeze()
        preds = torch.sigmoid(x).masked_fill(data_mask.logical_not(), 0)
        return (x, preds)

    def to_tensor_dict_module(self):
        return TensorDictModule(
            module=self,
            in_keys=['input', 'data_mask'],
            out_keys=['logits', 'preds']
        )
