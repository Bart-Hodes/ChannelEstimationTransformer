import torch
import torch.nn as nn

from models.Transformer.encoder import Encoder, EncoderBlock
from models.Transformer.decoder import Decoder, DecoderBlock, ProjectionLayer
from models.Transformer.embed import InputEmbeddings, PositionalEncoding
from models.Transformer.buildingblocks import (
    MultiHeadAttentionBlock,
    FeedForwardBlock,
)


class Transformer(nn.Module):
    """
    This class represents the Transformer model, which is a type of sequence-to-sequence model
    with encoder-decoder architecture. It uses self-attention mechanisms.

    Args:
        encoder (Encoder): The encoder part of the Transformer.
        decoder (Decoder): The decoder part of the Transformer.
        src_embed (InputEmbeddings): The input embeddings for the source language.
        tgt_embed (InputEmbeddings): The input embeddings for the target language.
        src_pos (PositionalEncoding): The positional encoding for the source language.
        tgt_pos (PositionalEncoding): The positional encoding for the target language.
        projection_layer (ProjectionLayer): The projection layer that maps the output to the target vocabulary size.

    Methods:
        encode(src, src_mask): Encodes the source sequence.
        decode(encoder_output, src_mask, tgt, tgt_mask): Decodes the encoded source sequence.
        project(x): Applies the projection layer to the output of the decoder.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
        pred_len,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        self.pred_len = pred_len

    def encode(self, src, src_mask=None):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)

    def forward(self, encoder_input, decoder_input):

        # Run the tensors through the encoder, decoder and the projection layer
        encoder_output = self.encode(encoder_input)  # (B, seq_len, d_model)
        decoder_output = self.decode(
            encoder_output, decoder_input
        )  # (B, seq_len, d_model)
        proj_output = self.project(decoder_output)  # (B, seq_len, vocab_size)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        return proj_output[:, -self.pred_len :, :]  # [B, L, D]


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    label_len: int,
    d_model: int = 512,
    N: int = 8,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    """
    This function builds a Transformer model with the specified parameters.

    Args:
        src_vocab_size (int): The size of the source vocabulary.
        tgt_vocab_size (int): The size of the target vocabulary.
        src_seq_len (int): The maximum length of the source sequences.
        tgt_seq_len (int): The maximum length of the target sequences.
        d_model (int, optional): The number of features in the input and output. Defaults to 512.
        N (int, optional): The number of layers in the encoder and decoder. Defaults to 6.

    Returns:
        Transformer: A Transformer model with the specified parameters.
    """

    # Create the embedding layers
    src_embed = InputEmbeddings(c_in=src_vocab_size, d_model=d_model)
    tgt_embed = InputEmbeddings(c_in=tgt_vocab_size, d_model=d_model)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len + label_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            d_model, encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            d_model,
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(
        encoder,
        decoder,
        src_embed,
        tgt_embed,
        src_pos,
        tgt_pos,
        projection_layer,
        tgt_seq_len,
    )

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
