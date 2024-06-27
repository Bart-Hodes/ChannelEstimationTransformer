import torch
import torch.nn as nn
import math

from TransformerModel.buildingblocks import (
    MultiHeadAttentionBlock,
    FeedForwardBlock,
    ResidualConnection,
    LayerNormalization,
)


class EncoderBlock(nn.Module):
    """
    This class is used to construct one encoder block in a Transformer model.

    Args:
        features (int): The number of features in the input and output.
        self_attention_block (MultiHeadAttentionBlock): The multi-head attention block to use.
        feed_forward_block (FeedForwardBlock): The feed forward block to use.
        dropout (float): The dropout rate.

    Returns:
        The output of the encoder block.
    """

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask, first_inference=True):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask, first_inference)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """
    This class is used to construct the encoder part of a Transformer model.

    Args:
        features (int): The number of features in the input and output.
        layers (nn.ModuleList): The list of layers to use in the encoder.

    Returns:
        The output of the encoder.
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask, first_inference=True):
        for layer in self.layers:
            x = layer(x, mask, first_inference=first_inference)
        return self.norm(x)
