import torch
import torch.nn as nn

from buildingblocks import (
    FeedForwardBlock,
    LayerNormalization,
)

from embed import DataEmbedding


class StreamingTransformer(nn.Module):

    def __init__(
        self,
        enc_in,
        numofblocks,
        total_seq_len,
        d_model,
        n_heads,
        d_ff,
        n_layers,
        dropout,
    ):
        super(StreamingTransformer, self).__init__()

        assert d_model % n_heads == 0
        assert total_seq_len % numofblocks == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout = dropout
        self.numofblocks = numofblocks
        self.total_seq_len = total_seq_len
        self.block_len = total_seq_len // numofblocks

        # Create the embedding layer
        self.embedding = DataEmbedding(enc_in, d_model, self.block_len)
        self.last_blocks = []

    def forward(self, x):
        # x [B, L, D]
        x = self.embedding(x)

        # Update the last_blocks list
        self.last_blocks.append(x[:, -self.numofblocks :, :])
        if len(self.last_blocks) >= 5:
            self.last_blocks = self.last_blocks[-5:]
        else:
            print(f"{len(self.last_blocks)} Not enough blocks to start processing")

        return x
