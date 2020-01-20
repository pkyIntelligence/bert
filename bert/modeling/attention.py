import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    :param q: query shape == (..., seq_len_q, depth)
    :param k: key shape == (..., seq_len_k, depth)
    :param v: value shape == (..., seq_len_v, depth_v)
    :param mask: Float tensor with shape broadcastable
                 to (..., seq_len_q, seq_len_k). Defaults to None.

    :return:
        output, attention_weights
    """

    rank_k = len(k.shape)
    matmul_qk = torch.matmul(q, k.transpose(rank_k-2, rank_k-1))

    # scale td_qk
    dk = torch.FloatTensor([k.shape[-1]])
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # (..., seq_len_q, seq_len_k)

    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_input, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(in_features=d_input, out_features=d_model)
        self.wk = nn.Linear(in_features=d_input, out_features=d_model)
        self.wv = nn.Linear(in_features=d_input, out_features=d_model)

        self.out_linear = nn.Linear(in_features=d_model, out_features=d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask):
        batch_size = q.shape[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)  # (batch_size, seq_len_q, d_model)

        output = self.out_linear(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
