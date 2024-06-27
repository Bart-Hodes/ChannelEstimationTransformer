import torch
import torch.nn as nn
import math

from TransformerModel.buildingblocks import (
    MultiHeadAttentionBlock,
    FeedForwardBlock,
    ResidualConnection,
    LayerNormalization,
)


class DecoderBlock(nn.Module):
    """
    Decoder block of the Transformer model.

    Args:
        features (int): The number of input features.
        self_attention_block (MultiHeadAttentionBlock): The self-attention block.
        cross_attention_block (MultiHeadAttentionBlock): The cross-attention block.
        feed_forward_block (FeedForwardBlock): The feed-forward block.
        dropout (float): The dropout rate.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): The self-attention block.
        cross_attention_block (MultiHeadAttentionBlock): The cross-attention block.
        feed_forward_block (FeedForwardBlock): The feed-forward block.
        residual_connections (nn.ModuleList): List of residual connections.

    """

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass of the DecoderBlock.

        Args:
            x: The input tensor.
            encoder_output: The output from the encoder.
            src_mask: The mask for the source sequence.
            tgt_mask: The mask for the target sequence.

        Returns:
            The output tensor after passing through the DecoderBlock.

        """
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    """
    Decoder module of the Transformer model.

    Args:
        features (int): The number of input features.
        layers (nn.ModuleList): List of decoder layers.

    Attributes:
        layers (nn.ModuleList): List of decoder layers.
        norm (LayerNormalization): Layer normalization module.

    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass of the decoder module.

        Args:
            x: The input tensor.
            encoder_output: The output tensor from the encoder.
            src_mask: The mask for the source sequence.
            tgt_mask: The mask for the target sequence.

        Returns:
            The output tensor after passing through the decoder layers and normalization.

        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """
    A projection layer that maps the input tensor to a tensor of the specified vocabulary size.

    Args:
        d_model (int): The size of the input tensor.
        vocab_size (int): The size of the vocabulary.

    Attributes:
        proj (nn.Linear): The linear transformation layer.

    """

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        """
        Forward pass of the projection layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The projected tensor.

        """
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
