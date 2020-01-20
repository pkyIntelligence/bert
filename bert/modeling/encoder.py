# Standard library imports
import torch
import torch.nn as nn

# Local application imports
from .attention import MultiHeadAttention
from .pos_encode import positional_encoding


def point_wise_feed_forward_network(d_input, d_model, dff):
    return nn.Sequential(
        nn.Linear(d_input, dff),
        nn.ReLU(),
        nn.Linear(dff, d_model)
    )


class EncoderBlock(nn.Module):
    def __init__(self, input_seq_len, d_input, d_model, num_heads, dff, rate=0.1):
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(d_input, d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, d_model, dff)

        self.layernorm1 = nn.LayerNorm(normalized_shape=[input_seq_len, d_model], eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=[input_seq_len, d_model], eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class Encoder(nn.Module):
    def __init__(self, num_blocks, input_seq_len, d_model, num_heads, dff, input_vocab_size,
                 max_pos_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_blocks = num_blocks

        self.embedding = nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=d_model)
        self.pos_encoding = positional_encoding(max_pos_encoding, self.d_model)

        self.enc_blocks = [EncoderBlock(input_seq_len, d_model, d_model, num_heads, dff, rate)
                           for _ in range(num_blocks)]

        self.dropout = nn.Dropout(rate)

    def forward(self, x, mask):

        seq_len = x.shape[1]

        # adding embedding and position encoding
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= torch.sqrt(torch.Tensor([self.d_model]))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_blocks):
            x = self.enc_blocks[i](x, mask)

        return x  # (batch_size, input_seq_len, d_model)
