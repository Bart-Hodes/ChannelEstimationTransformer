import torch
import torch.nn as nn
import math

import scipy


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 10**-6) -> None:
        """
        LayerNormalization is a module that applies layer normalisation to the input tensor.

        Args:
            features (int): The number of features in the module.
            eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 10**-6.
        """
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(
            torch.ones(features)
        )  # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features))  # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """
    FeedForwardBlock is a module that applies a feed-forward neural network to the input tensor.

    Args:
        d_model (int): The number of expected features in the input tensor.
        d_ff (int): The number of output features in the feed-forward neural network.
        dropout (float): The dropout probability.

    Attributes:
        linear_1 (nn.Linear): The first linear layer of the feed-forward neural network.
        dropout (nn.Dropout): The dropout layer.
        linear_2 (nn.Linear): The second linear layer of the feed-forward neural network.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        """
        Forward pass of the FeedForwardBlock.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the feed-forward neural network.
        """
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    """
    This class is used to apply multi-head attention to the input data.

    Parameters:
    -----------
    d_model : int
        The embedding vector size.
    h : int
        The number of heads.
    dropout : float
        The dropout rate to apply.

    Attributes:
    -----------
    d_model : int
        The embedding vector size.
    h : int
        The number of heads.
    d_k : int
        The dimension of the vector seen by each head.
    w_q : nn.Linear
        The linear layer for the query projection.
    w_k : nn.Linear
        The linear layer for the key projection.
    w_v : nn.Linear
        The linear layer for the value projection.
    w_o : nn.Linear
        The linear layer for the output projection.
    dropout : nn.Dropout
        The dropout layer to apply.
    """

    def __init__(
        self,
        d_model: int,
        h: int,
        dropout: float,
        debug_layer_index: int,
        shift_size: int = 0,
    ) -> None:
        super().__init__()

        self.debug_layer_index = debug_layer_index

        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h  # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv

        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

        self.prev_attn = None
        self.shift_size = shift_size

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Calculate attention scores according to the formula in the paper.

        parameters:
        -----------
        query: torch.Tensor
            Query tensor of shape (batch, h, seq_len, d_k)
        key: torch.Tensor
            Key tensor of shape (batch, h, seq_len, d_k)
        value: torch.Tensor
            Value tensor of shape (batch, h, seq_len, d_k)
        mask: torch.Tensor
            Mask tensor of shape (batch, seq_len, seq_len)
        dropout: nn.Dropout
            Dropout layer to apply to the attention scores
        """

        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        attention_debug = attention_scores

        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(
            dim=-1
        )  # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores, attention_debug

    def forward(self, q, k, v, mask, first_inference=True):
        """
        Forward pass for the multi-head attention block.

        parameters:
        -----------
        q: torch.Tensor
            Query tensor of shape (batch, seq_len, d_model)
        k: torch.Tensor
            Key tensor of shape (batch, seq_len, d_model)
        v: torch.Tensor
            Value tensor of shape (batch, seq_len, d_model)
        mask: torch.Tensor
            Mask tensor of shape (batch, seq_len, seq_len)
        prev_attn: torch.Tensor
            Previous attention scores to reuse (optional)
        shift_size: int
            The shift size for the sliding window (optional)
        """
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # Split the d_model into h heads
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        # print(f"First Inference: {first_inference}, Shift Size: {self.shift_size}")
        if first_inference == False and self.shift_size > 0:
            new_query = query[:, :, -self.shift_size :]
            new_key = key[:, :, -self.shift_size :]
            new_value = value[:, :, -self.shift_size :]

            __, __, seq_len, d_model = query.shape

            print(
                f"New Query Shape: {new_query.shape},New Key Shape: {new_key.shape},New Value Shape: {new_value.shape}"
            )
            print(
                f"Query Shape: {query.shape},Key Shape: {key.shape},Value Shape: {value.shape}"
            )

            print(f"attention_scores_shape: {self.attention_scores.shape}")
            rightmost_column = torch.zeros(seq_len)
            for i in range(seq_len):
                rightmost_column[i] = sum(
                    query[:, :, i, k] * key[:, :, k, d_model - 1]
                    for k in range(d_model)
                )

            bottommost_row = torch.zeros(seq_len)
            for j in range(seq_len):
                bottommost_row[j] = sum(
                    query[:, :, seq_len - 1, k] * key[:, :, k, j]
                    for k in range(d_model)
                )

            x = self.attention_scores @ value

            (
                x_full_precision,
                attention_scores_full_precision,
                attention_debug_full_precision,
            ) = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

            import matplotlib.pyplot as plt

            plt.imshow(attention_debug_full_precision[0, 0, :, :].detach().numpy())
            plt.colorbar()  # Show color scale
            plt.title("Heatmap Example")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.savefig("attention_debug_full_precision.png", dpi=300)

        else:
            x, self.attention_scores, self.attention_debug = (
                MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
            )

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    """
    A residual connection module that applies residual connection to the input tensor.

    Args:
        features (int): The number of input features.
        dropout (float): The dropout probability.

    Attributes:
        dropout (nn.Dropout): The dropout layer.
        norm (LayerNormalization): The layer normalization module.

    """

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        """
        Apply the residual connection to the input tensor.

        Args:
            x: The input tensor.
            sublayer: The sublayer function.

        Returns:
            The output tensor after applying the residual connection.

        """
        return x + self.dropout(sublayer(self.norm(x)))
