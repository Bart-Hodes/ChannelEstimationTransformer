import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class PositionalEncoding(nn.Module):
    """
    This class is used to embed the positional information of the input data. This is done by creating a
    vector of size d_model for each position in the input data.

    Method:
    -------
    __init__(d_model, max_len=5000)
        Initialize the PositionalEncoding class.

    forward(x)
        Apply positional embedding to input data.
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len, 1)
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

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(
            False
        )  # (batch, seq_len, d_model)
        return self.dropout(x)


class InputEmbeddings(nn.Module):
    """
    This class is used to embed the input tokens. Since we are working with continuous data,
    we use a 1D convolution instead of the typical token embedding layer. To this extend, we
    perform a convolution over the Nr*Nt*2 dimension of the input data.

    Method:
    -------
    __init__(c_in, d_model)
        Initialize the InputEmbeddings class.

    forward(x)
        Apply input embedding to input data.
    """

    def __init__(self, c_in: int, d_model: int):
        """
        Parameters
        ----------
        c_in : int
            Number of input channels.
        d_model : int
            Dimensionality of the model.
        """

        super(InputEmbeddings, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenEmbedding = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
        )
        # self.tokenEmbedding = nn.Linear(c_in, d_model)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        # Reshape input to (batch, Nr*Nt*2, OFDMSlot) and apply 1D convolution
        # x = self.tokenEmbedding(x)
        x = self.tokenEmbedding(x.permute(0, 2, 1)).transpose(1, 2)
        # Reshape back to (batch, Nr*Nt*2, d_model)
        return x
