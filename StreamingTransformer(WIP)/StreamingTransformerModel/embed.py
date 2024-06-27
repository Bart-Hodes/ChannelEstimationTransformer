import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    # This class is used to add positional encoding to the input of the Transformer model. However, we would
    # like to use a positional encoding over a time sequence. Therefore we add an position counter to see where
    # we are in the sequence.

    # TODO look at not making a fixed size encoding book.

    def __init__(self, d_model: int, dropout: float) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = 1000
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(self.seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len,1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer("pe", pe)

    def forward(self, x, position_counter):
        x = x + (
            self.pe[:, position_counter : position_counter + x.shape[1], :]
        ).requires_grad_(
            False
        )  # (batch, seq_len, d_model)
        return self.dropout(x)


class InputEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(InputEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.position_counter = 0

        self.input_embedding = InputEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.position_embedding(x, self.position_counter)

        # Increment position counter with the length of the input
        self.position_counter += x.shape[1]
        return self.dropout(x)
